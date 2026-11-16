// Entraînement autonome Mini-GPT (Candle 0.8), en mode terminal.
// Lancez depuis la racine du projet pour que les chemins de données et checkpoints
// soient résolus relativement au répertoire courant.

use candle_core::{backprop::GradStore, DType, Device, Tensor, D, Var};
use candle_nn::{
    embedding, layer_norm, linear, loss, ops, AdamW, Embedding, LayerNorm, Linear, Module,
    Optimizer, ParamsAdamW, VarBuilder, VarMap,
};
use rand::Rng;
use std::path::Path;

const VOCAB: usize = 256; // tokenizer byte-level : un octet UTF-8 par token
const CONTEXTE: usize = 256;
const BATCH: usize = 64;
const LR_MAX: f64 = 1e-3;
const DIM_MODELE: usize = 256;
const NB_TETES: usize = 4;
const NB_BLOCS: usize = 4;
const FICHIER_POIDS: &str = "transformer_code.safetensors";
const FICHIER_DERNIER: &str = "transformer_code_last.safetensors";
const EVAL_PERIODIQUE: usize = 100;
const SAUVEGARDE_PERIODIQUE: usize = 500;
const NORME_GRAD_MAX: f64 = 1.0;
const NB_PAS_DEFAUT: usize = 60000;
const DONNEES_DEFAUT: &str = "donnees.txt;langague.txt";

type Res<T> = std::result::Result<T, Box<dyn std::error::Error>>;
type CRes<T> = candle_core::Result<T>;

struct CausalSelfAttention {
    qkv: Linear,
    proj: Linear,
}

impl CausalSelfAttention {
    fn new(vb: VarBuilder) -> CRes<Self> {
        let qkv = linear(DIM_MODELE, DIM_MODELE * 3, vb.pp("qkv"))?;
        let proj = linear(DIM_MODELE, DIM_MODELE, vb.pp("proj"))?;
        Ok(Self { qkv, proj })
    }

    fn forward(&self, x: &Tensor) -> CRes<Tensor> {
        let (b, t, c) = x.dims3()?;
        let hd = c / NB_TETES;
        let qkv = self
            .qkv
            .forward(x)?
            .reshape((b, t, 3, NB_TETES, hd))?
            .permute((2, 0, 3, 1, 4))?;
        let q = qkv.get(0)?.contiguous()?;
        let k = qkv.get(1)?.contiguous()?;
        let v = qkv.get(2)?.contiguous()?;
        let att = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? / (hd as f64).sqrt())?;

        // Masque causal répliqué pour chaque tête (évite le mismatch de broadcast).
        let mut m = vec![0f32; NB_TETES * t * t];
        for tete in 0..NB_TETES {
            let decalage = tete * t * t;
            for i in 0..t {
                for j in (i + 1)..t {
                    m[decalage + i * t + j] = f32::NEG_INFINITY;
                }
            }
        }
        let masque = Tensor::from_vec(m, (1, NB_TETES, t, t), x.device())?.to_dtype(x.dtype())?;
        let att = ops::softmax(&att.broadcast_add(&masque)?, D::Minus1)?;
        let out = att
            .matmul(&v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, t, c))?;
        self.proj.forward(&out)
    }
}

struct BlocTransformer {
    ln1: LayerNorm,
    attn: CausalSelfAttention,
    ln2: LayerNorm,
    mlp_up: Linear,
    mlp_down: Linear,
}

impl BlocTransformer {
    fn new(vb: VarBuilder) -> CRes<Self> {
        Ok(Self {
            ln1: layer_norm(DIM_MODELE, 1e-5, vb.pp("ln1"))?,
            attn: CausalSelfAttention::new(vb.pp("attn"))?,
            ln2: layer_norm(DIM_MODELE, 1e-5, vb.pp("ln2"))?,
            mlp_up: linear(DIM_MODELE, DIM_MODELE * 4, vb.pp("up"))?,
            mlp_down: linear(DIM_MODELE * 4, DIM_MODELE, vb.pp("down"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> CRes<Tensor> {
        let x = (x + self.attn.forward(&self.ln1.forward(x)?)?)?;
        let h = self.mlp_up.forward(&self.ln2.forward(&x)?)?.gelu()?;
        &x + self.mlp_down.forward(&h)?
    }
}

struct TransformerCodeIA {
    tok_emb: Embedding,
    pos_emb: Embedding,
    blocs: Vec<BlocTransformer>,
    ln_f: LayerNorm,
    tete: Linear,
}

impl TransformerCodeIA {
    fn new(vb: VarBuilder) -> CRes<Self> {
        let tok_emb = embedding(VOCAB, DIM_MODELE, vb.pp("tok_emb"))?;
        let pos_emb = embedding(CONTEXTE, DIM_MODELE, vb.pp("pos_emb"))?;
        let mut blocs = Vec::new();
        for i in 0..NB_BLOCS {
            blocs.push(BlocTransformer::new(vb.pp(format!("b{i}")))?);
        }
        let ln_f = layer_norm(DIM_MODELE, 1e-5, vb.pp("ln_f"))?;
        let tete = linear(DIM_MODELE, VOCAB, vb.pp("tete"))?;
        Ok(Self { tok_emb, pos_emb, blocs, ln_f, tete })
    }

    fn forward(&self, idx: &Tensor) -> CRes<Tensor> {
        let (_b, t) = idx.dims2()?;
        let pos = Tensor::arange(0u32, t as u32, idx.device())?.unsqueeze(0)?;
        let mut x = self
            .tok_emb
            .forward(idx)?
            .broadcast_add(&self.pos_emb.forward(&pos)?)?;
        for bloc in &self.blocs {
            x = bloc.forward(&x)?;
        }
        let x = self.ln_f.forward(&x)?;
        self.tete.forward(&x)
    }
}

struct Moteur {
    modele: TransformerCodeIA,
    varmap: VarMap,
    device: Device,
}

impl Moteur {
    fn new() -> CRes<Self> {
        #[cfg(feature = "cuda")]
        let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
        #[cfg(not(feature = "cuda"))]
        let device = Device::Cpu;
        println!("Device sélectionné : {:?}", device);

        let dtype = if matches!(&device, Device::Cuda(_)) { DType::BF16 } else { DType::F32 };
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);
        let modele = TransformerCodeIA::new(vb)?;
        Ok(Self { modele, varmap, device })
    }

    fn sauver(&self, chemin: &str) -> CRes<()> {
        self.varmap.save(chemin)
    }

    fn sauver_atomique(&self, chemin: &str) -> Res<()> {
        let temporaire = format!("{chemin}.tmp");
        self.sauver(&temporaire)?;
        std::fs::rename(&temporaire, chemin)?;
        Ok(())
    }

    fn charger(&mut self, chemin: &str) -> CRes<()> {
        self.varmap.load(chemin)
    }
}

struct Source {
    train: Tensor,
    val: Tensor,
}

fn chemins(s: &str) -> Vec<String> {
    s.split(';')
        .map(|c| c.trim().to_string())
        .filter(|c| !c.is_empty())
        .collect()
}

fn tirer_lot(
    sources: &[Source],
    validation: bool,
    batch: usize,
    long: usize,
    rng: &mut impl Rng,
) -> CRes<(Tensor, Tensor)> {
    if sources.is_empty() || batch == 0 {
        return Err(candle_core::Error::Msg("lot vide ou aucune source de données".into()));
    }
    let long = sources
        .iter()
        .map(|s| {
            let data = if validation { &s.val } else { &s.train };
            data.dims1().map(|n| n.saturating_sub(1))
        })
        .collect::<CRes<Vec<_>>>()?
        .into_iter()
        .min()
        .unwrap_or(0)
        .min(long)
        .max(1);

    let mut sequences = Vec::with_capacity(batch);
    for _ in 0..batch {
        let source = &sources[rng.gen_range(0..sources.len())];
        let data = if validation { &source.val } else { &source.train };
        let n = data.dims1()?;
        if n <= long {
            return Err(candle_core::Error::Msg(
                "split de données trop court pour former un exemple".into(),
            ));
        }
        let debut = rng.gen_range(0..(n - long));
        sequences.push(data.narrow(0, debut, long + 1)?.unsqueeze(0)?);
    }
    let refs: Vec<&Tensor> = sequences.iter().collect();
    let tokens = Tensor::cat(&refs, 0)?;
    let x = tokens.narrow(1, 0, long)?;
    let y = tokens.narrow(1, 1, long)?.reshape((batch * long,))?;
    Ok((x, y))
}

fn clip_grad_norm(grads: &mut GradStore, vars: &[Var], max_norm: f64) -> Res<f64> {
    let mut sum_sq: Option<Tensor> = None;
    for var in vars {
        if let Some(g) = grads.get(var) {
            let part = g.to_dtype(DType::F32)?.sqr()?.sum_all()?;
            sum_sq = Some(match sum_sq {
                Some(acc) => (&acc + &part)?,
                None => part,
            });
        }
    }
    let norm = match sum_sq {
        Some(sum) => sum.to_scalar::<f32>()? as f64,
        None => 0.0,
    }
    .sqrt();
    if !norm.is_finite() {
        return Err("NaN/Inf détecté dans la norme des gradients".into());
    }
    if norm > max_norm {
        let facteur = max_norm / (norm + 1e-12);
        for var in vars {
            if let Some(g) = grads.remove(var) {
                grads.insert(var, (&g * facteur)?);
            }
        }
    }
    Ok(norm)
}

fn evaluer_loss(
    moteur: &Moteur,
    sources: &[Source],
    batch: usize,
    long: usize,
    rng: &mut impl Rng,
) -> Res<f32> {
    let (x, y) = tirer_lot(sources, true, batch, long, rng)?;
    let (b, t) = x.dims2()?;
    let logits = moteur
        .modele
        .forward(&x)?
        .reshape((b * t, VOCAB))?
        .to_dtype(DType::F32)?;
    let valeur = loss::cross_entropy(&logits, &y)?.to_scalar::<f32>()?;
    if !valeur.is_finite() {
        return Err("NaN/Inf détecté dans la loss de validation".into());
    }
    Ok(valeur)
}

fn entrainement(moteur: &mut Moteur, fichiers: &str, pas_total: usize) -> Res<()> {
    let long = CONTEXTE.clamp(16, CONTEXTE);
    let liste = chemins(fichiers);
    if liste.is_empty() {
        return Err("aucun fichier de données indiqué".into());
    }

    let mut sources = Vec::new();
    for chemin in &liste {
        let data = std::fs::read(chemin).map_err(|e| format!("'{chemin}' : {e}"))?;
        if data.len() < 20 {
            return Err(format!("'{chemin}' est trop petit (minimum 20 octets)").into());
        }
        let coupe = (data.len() * 90 / 100).clamp(2, data.len() - 2);
        let tokens: Vec<u32> = data.iter().map(|&byte| byte as u32).collect();
        let tous = Tensor::from_vec(tokens, data.len(), &moteur.device)?;
        let train = tous.narrow(0, 0, coupe)?;
        let val = tous.narrow(0, coupe, data.len() - coupe)?;
        println!(
            "{chemin} : {:.1} Mo chargés (90 % train / 10 % validation)",
            data.len() as f64 / 1e6
        );
        sources.push(Source { train, val });
    }
    println!("Entraînement sur {} fichier(s), tirés à parts égales.", sources.len());

    let reprise = if Path::new(FICHIER_DERNIER).exists() {
        Some(FICHIER_DERNIER)
    } else if Path::new(FICHIER_POIDS).exists() {
        Some(FICHIER_POIDS)
    } else {
        None
    };
    if let Some(chemin) = reprise {
        match moteur.charger(chemin) {
            Ok(()) => println!("Reprise des poids depuis {chemin} (état AdamW réinitialisé)."),
            Err(e) => eprintln!("Checkpoint {chemin} illisible ({e}); démarrage avec des poids neufs."),
        }
    } else {
        println!("Aucun checkpoint existant : initialisation de poids neufs.");
    }

    let vars = moteur.varmap.all_vars();
    let mut opt = AdamW::new(
        vars.clone(),
        ParamsAdamW { lr: LR_MAX, weight_decay: 0.01, beta2: 0.95, ..Default::default() },
    )?;
    let mut rng = rand::thread_rng();
    let chaud = 300usize.min(pas_total.saturating_sub(1)).max(1);
    let lr_min = LR_MAX * 0.1;

    let mut moyenne_initiale = 0.0f64;
    for _ in 0..4 {
        moyenne_initiale += evaluer_loss(moteur, &sources, BATCH.min(8).max(1), long, &mut rng)? as f64;
    }
    let mut meilleure_val = (moyenne_initiale / 4.0) as f32;
    if !meilleure_val.is_finite() {
        return Err("NaN/Inf détecté dans la loss de validation initiale".into());
    }
    moteur.sauver_atomique(FICHIER_POIDS)?;
    println!("Loss validation initiale {meilleure_val:.4}; modèle sauvegardé dans {FICHIER_POIDS}.");

    for pas in 1..=pas_total {
        let lr = if pas <= chaud {
            LR_MAX * pas as f64 / chaud as f64
        } else {
            let prog = (pas - chaud) as f64 / (pas_total - chaud).max(1) as f64;
            lr_min + 0.5 * (LR_MAX - lr_min) * (1.0 + (std::f64::consts::PI * prog.min(1.0)).cos())
        };
        opt.set_learning_rate(lr);

        let (x, y) = tirer_lot(&sources, false, BATCH, long, &mut rng)?;
        let (b, t) = x.dims2()?;
        let logits = moteur
            .modele
            .forward(&x)?
            .reshape((b * t, VOCAB))?
            .to_dtype(DType::F32)?;
        let perte = loss::cross_entropy(&logits, &y)?;
        let valeur = if pas % 50 == 0 { Some(perte.to_scalar::<f32>()?) } else { None };
        if valeur.is_some_and(|v| !v.is_finite()) {
            return Err(format!("NaN/Inf dans la loss d'entraînement au pas {pas}").into());
        }
        let mut grads = perte.backward()?;
        let norme_grad = clip_grad_norm(&mut grads, &vars, NORME_GRAD_MAX)?;
        opt.step(&grads)?;

        let mut val_perte = None;
        if pas == 1 || pas % EVAL_PERIODIQUE == 0 || pas == pas_total {
            let mut moyenne = 0.0f64;
            for _ in 0..4 {
                moyenne += evaluer_loss(moteur, &sources, BATCH.min(8).max(1), long, &mut rng)? as f64;
            }
            let v = (moyenne / 4.0) as f32;
            if !v.is_finite() {
                return Err(format!("NaN/Inf dans la loss de validation au pas {pas}").into());
            }
            val_perte = Some(v);
            if v < meilleure_val {
                meilleure_val = v;
                moteur.sauver_atomique(FICHIER_POIDS)?;
                println!("Nouveau meilleur modèle : validation {v:.4} ({FICHIER_POIDS}).");
            }
        }

        if pas % 50 == 0 || val_perte.is_some() {
            println!(
                "pas {pas}/{pas_total} - perte {} - validation {} - lr {lr:.2e} - norme grad {norme_grad:.3}",
                valeur.map(|v| format!("{v:.4}")).unwrap_or_else(|| "--".into()),
                val_perte.map(|v| format!("{v:.4}")).unwrap_or_else(|| "--".into())
            );
        }
        if pas % SAUVEGARDE_PERIODIQUE == 0 {
            moteur.sauver_atomique(FICHIER_DERNIER)?;
            println!("Checkpoint de reprise sauvegardé au pas {pas} : {FICHIER_DERNIER}.");
        }
    }

    moteur.sauver_atomique(FICHIER_DERNIER)?;
    println!(
        "Entraînement terminé. Checkpoint reprise : {FICHIER_DERNIER}; meilleur modèle : {FICHIER_POIDS} (loss validation {meilleure_val:.4})."
    );
    Ok(())
}

fn main() -> Res<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let fichiers = args.first().map(String::as_str).unwrap_or(DONNEES_DEFAUT);
    let pas = args
        .get(1)
        .map(|s| s.parse::<usize>())
        .transpose()?
        .unwrap_or(NB_PAS_DEFAUT);
    println!("Données : {fichiers}; nombre de pas : {pas}; contexte : {CONTEXTE}; batch : {BATCH}.");
    let mut moteur = Moteur::new()?;
    entrainement(&mut moteur, fichiers, pas)
}