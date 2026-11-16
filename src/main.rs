// CUDA (Cargo.toml) :
// candle-core = { version = "0.8", features = ["cuda"] }
// candle-nn   = { version = "0.8", features = ["cuda"] }
// Sans CUDA toolkit installé, retirer la feature ["cuda"] de ces deux dépendances.
// =====================================================================
//  STUDIO IA CODE - 100% Rust (candle + egui)
//  Trois parties dans une seule application :
//    1. Entraînement : entraîne le modèle sur donnees.txt (thread séparé)
//    2. Discussion   : chat avec le modèle (format <|user|> / <|assistant|>)
//    3. Création     : génération de code (complétion ou depuis une description)
// =====================================================================

use candle_core::{backprop::GradStore, DType, Device, Tensor, D, Var};
use candle_nn::{
    embedding, layer_norm, linear, loss, ops, AdamW, Embedding, LayerNorm, Linear, Module,
    Optimizer, ParamsAdamW, VarBuilder, VarMap,
};
use eframe::egui;
use rand::distributions::{Distribution, WeightedIndex};
use rand::Rng;
use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

type Res<T> = std::result::Result<T, Box<dyn std::error::Error>>;
type CRes<T> = candle_core::Result<T>;

// --- Hyperparamètres du modèle (si tu les changes, supprime l'ancien .safetensors) ---
const VOCAB: usize = 256; // 1 token = 1 octet
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

fn verrou<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

// =====================================================================
//  ARCHITECTURE TRANSFORMER (MINI-GPT)
// =====================================================================
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

        // Masque causal : un token ne voit pas le futur
        let mut m = vec![0f32; t * t];
        for i in 0..t {
            for j in (i + 1)..t {
                m[i * t + j] = f32::NEG_INFINITY;
            }
        }
        let masque = Tensor::from_vec(m, (1, 1, t, t), x.device())?.to_dtype(x.dtype())?;
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

// Modèle + poids, partagé entre l'interface et les threads
struct Moteur {
    modele: TransformerCodeIA,
    varmap: VarMap,
    device: Device,
}

impl Moteur {
    fn new() -> CRes<Self> {
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        println!("Device sélectionné : {:?}", device);
        let dtype = if matches!(&device, Device::Cuda(_)) { DType::BF16 } else { DType::F32 };
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);
        let modele = TransformerCodeIA::new(vb)?;
        Ok(Self { modele, varmap, device })
    }
    fn nb_params(&self) -> usize {
        self.varmap.all_vars().iter().map(|v| v.elem_count()).sum()
    }
    fn sauver(&self, chemin: &str) -> CRes<()> {
        self.varmap.save(chemin)
    }
    fn charger(&mut self, chemin: &str) -> CRes<()> {
        self.varmap.load(chemin)
    }
}

// =====================================================================
//  GÉNÉRATION (discussion + création), dans un thread
// =====================================================================
#[derive(Default)]
struct Gen {
    texte: String,
    en_cours: bool,
    annuler: bool,
    fini_non_lu: bool,
}

fn prochain_token(
    m: &Moteur,
    fenetre: &[u32],
    temperature: f64,
    top_k: usize,
    rng: &mut impl Rng,
) -> CRes<u32> {
    let len = fenetre.len();
    let entree = Tensor::from_vec(fenetre.to_vec(), (1, len), &m.device)?;
    let logits = m.modele.forward(&entree)?;
    let l: Vec<f32> = logits
        .to_dtype(DType::F32)?
        .narrow(1, len - 1, 1)?
        .squeeze(1)?
        .squeeze(0)?
        .to_vec1()?;
    if l.iter().any(|v| !v.is_finite()) {
        return Err(candle_core::Error::Msg("NaN/Inf détecté dans les logits de génération".into()));
    }

    let mut idx: Vec<usize> = (0..l.len()).collect();
    idx.sort_by(|a, b| l[*b].partial_cmp(&l[*a]).unwrap_or(std::cmp::Ordering::Equal));
    idx.truncate(top_k.clamp(1, VOCAB));
    let max = l[idx[0]];
    let t = temperature.max(0.05) as f32;
    let poids: Vec<f32> = idx.iter().map(|&i| ((l[i] - max) / t).exp()).collect();
    match WeightedIndex::new(&poids) {
        Ok(dist) => Ok(idx[dist.sample(rng)] as u32),
        Err(_) => Ok(idx[0] as u32),
    }
}

fn lancer_generation(
    moteur: Arc<Mutex<Moteur>>,
    gen: Arc<Mutex<Gen>>,
    prompt: String,
    max_octets: usize,
    temperature: f64,
    top_k: usize,
    format_chat: bool,
) {
    {
        let mut g = verrou(&gen);
        g.texte.clear();
        g.en_cours = true;
        g.annuler = false;
        g.fini_non_lu = false;
    }
    std::thread::spawn(move || {
        let mut tokens: Vec<u32> = prompt.bytes().map(|b| b as u32).collect();
        if tokens.is_empty() {
            tokens.push(b'\n' as u32);
        }
        let mut sortie: Vec<u8> = Vec::new();
        let mut rng = rand::thread_rng();
        let mut erreur: Option<String> = None;

        for _ in 0..max_octets {
            if verrou(&gen).annuler {
                break;
            }
            let debut = tokens.len().saturating_sub(CONTEXTE);
            let res = {
                let m = verrou(&moteur);
                prochain_token(&m, &tokens[debut..], temperature, top_k, &mut rng)
            };
            let id = match res {
                Ok(i) => i,
                Err(e) => {
                    erreur = Some(e.to_string());
                    break;
                }
            };
            tokens.push(id);
            sortie.push(id as u8);

            let mut texte = String::from_utf8_lossy(&sortie).to_string();
            let mut fin = false;
            if format_chat {
                for marque in ["<|end|>", "<|user|>"] {
                    if let Some(p) = texte.find(marque) {
                        texte.truncate(p);
                        fin = true;
                        break;
                    }
                }
            }
            if (sortie.len() % 8 == 0) || fin || sortie.len() == max_octets {
                verrou(&gen).texte = texte;
            }
            if fin {
                break;
            }
        }

        let mut g = verrou(&gen);
        if format_chat {
            g.texte = g.texte.trim().to_string();
        }
        if let Some(e) = erreur {
            g.texte.push_str(&format!("\n[erreur : {e}]"));
        }
        g.en_cours = false;
        g.fini_non_lu = true;
    });
}

// =====================================================================
//  ENTRAÎNEMENT, dans un thread
// =====================================================================
#[derive(Default)]
struct EtatTrain {
    en_cours: bool,
    stop: bool,
    pas: usize,
    total: usize,
    pertes: Vec<(usize, f32)>,
    val_pertes: Vec<(usize, f32)>,
    log: Vec<String>,
    debut: Option<Instant>,
}

#[derive(Clone)]
struct ParamsTrain {
    chemin: String,
    pas: usize,
    batch: usize,
    longueur: usize,
    lr: f64,
}

// Les octets des fichiers sont transférés une fois sur le device sous forme de tokens u32.
// train/val sont des vues (narrow) sur le Tensor original, pas des copies hôte.
struct Source {
    train: Tensor,
    val: Tensor,
}

impl Source {
    fn train(&self) -> &Tensor { &self.train }
    fn val(&self) -> &Tensor { &self.val }
}

// "donnees.txt;donnees_chat.txt" -> ["donnees.txt", "donnees_chat.txt"]
fn chemins(s: &str) -> Vec<String> {
    s.split(';')
        .map(|c| c.trim().to_string())
        .filter(|c| !c.is_empty())
        .collect()
}

// Chaque séquence du lot est une vue étroite sur les tokens préchargés du fichier choisi.
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
    let long = sources.iter()
        .map(|s| {
            let data = if validation { s.val() } else { s.train() };
            data.dims1().map(|n| n.saturating_sub(1))
        })
        .collect::<CRes<Vec<_>>>()?
        .into_iter().min().unwrap_or(0).min(long).max(1);
    let mut sequences = Vec::with_capacity(batch);
    for _ in 0..batch {
        let source = &sources[rng.gen_range(0..sources.len())];
        let data = if validation { source.val() } else { source.train() };
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

// Norme L2 globale sur tous les gradients, puis mise à l'échelle en place.
// candle 0.8 expose les gradients via GradStore::get/remove/insert.
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
    }.sqrt();
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
    moteur: &Arc<Mutex<Moteur>>,
    sources: &[Source],
    batch: usize,
    long: usize,
    rng: &mut impl Rng,
) -> Res<f32> {
    let (x, y) = tirer_lot(sources, true, batch, long, rng)?;
    let (b, t) = x.dims2()?;
    let m = verrou(moteur);
    let logits = m.modele.forward(&x)?.reshape((b * t, VOCAB))?.to_dtype(DType::F32)?;
    let valeur = loss::cross_entropy(&logits, &y)?.to_scalar::<f32>()?;
    if !valeur.is_finite() {
        return Err("NaN/Inf détecté dans la loss de validation".into());
    }
    Ok(valeur)
}

fn entrainement(moteur: &Arc<Mutex<Moteur>>, etat: &Arc<Mutex<EtatTrain>>, p: &ParamsTrain) -> Res<()> {
    let long = p.longueur.clamp(16, CONTEXTE);
    let liste = chemins(&p.chemin);
    if liste.is_empty() {
        return Err("aucun fichier de données indiqué".into());
    }
    let mut sources: Vec<Source> = Vec::new();
    let device = verrou(moteur).device.clone();
    for chemin in &liste {
        let data = std::fs::read(chemin).map_err(|e| format!("'{chemin}' : {e}"))?;
        if data.len() < 20 {
            return Err(format!("'{chemin}' est trop petit pour constituer train/validation (minimum 20 octets)").into());
        }
        let coupe = (data.len() * 90 / 100).clamp(2, data.len() - 2);
        let tokens: Vec<u32> = data.iter().map(|&byte| byte as u32).collect();
        let tous = Tensor::from_vec(tokens, data.len(), &device)?;
        let train = tous.narrow(0, 0, coupe)?;
        let val = tous.narrow(0, coupe, data.len() - coupe)?;
        verrou(etat)
            .log
            .push(format!("{chemin} : {:.1} Mo chargés sur le device (90 % train / 10 % validation)", data.len() as f64 / 1e6));
        sources.push(Source { train, val });
    }
    verrou(etat).log.push(format!(
        "Entraînement sur {} fichier(s), tirés à parts égales.",
        sources.len()
    ));

    // Reprise automatique du dernier checkpoint. Seuls les poids sont stockés;
    // l'état interne d'AdamW (moments) repart donc de zéro.
    if Path::new(FICHIER_DERNIER).exists() {
        match verrou(moteur).charger(FICHIER_DERNIER) {
            Ok(()) => verrou(etat).log.push(format!("Reprise des poids depuis {FICHIER_DERNIER}. État AdamW réinitialisé.")),
            Err(e) => verrou(etat).log.push(format!("Checkpoint last illisible ({e}); démarrage avec les poids courants.")),
        }
    }

    let vars = verrou(moteur).varmap.all_vars();
    let mut opt = AdamW::new(
        vars.clone(),
        ParamsAdamW { lr: p.lr, weight_decay: 0.01, beta2: 0.95, ..Default::default() },
    )?;
    let mut rng = rand::thread_rng();
    let chaud = 300usize.min(p.pas.saturating_sub(1)).max(1);
    let lr_min = p.lr * 0.1;
    // Référence de validation des poids chargés (utile lors d'une reprise),
    // afin de ne pas remplacer un meilleur modèle par un résultat moins bon.
    let mut moyenne_initiale = 0.0f64;
    for _ in 0..4 {
        moyenne_initiale += evaluer_loss(moteur, &sources, p.batch.min(8).max(1), long, &mut rng)? as f64;
    }
    let mut meilleure_val = (moyenne_initiale / 4.0) as f32;
    if !meilleure_val.is_finite() {
        return Err("NaN/Inf détecté dans la loss de validation initiale".into());
    }
    verrou(moteur).sauver(FICHIER_POIDS)?;
    verrou(etat).log.push(format!("Loss validation de référence {meilleure_val:.4}; poids sauvegardés dans {FICHIER_POIDS}."));
    {
        let mut e = verrou(etat);
        e.val_pertes.push((0, meilleure_val));
    }

    for pas in 1..=p.pas {
        if verrou(etat).stop {
            verrou(etat).log.push("Arrêt demandé; checkpoint last sauvegardé.".into());
            break;
        }
        // Warmup ~300 pas puis décroissance cosinus jusqu'à 10 % du LR maximal.
        let lr = if pas <= chaud {
            p.lr * pas as f64 / chaud as f64
        } else {
            let prog = (pas - chaud) as f64 / (p.pas - chaud).max(1) as f64;
            lr_min + 0.5 * (p.lr - lr_min) * (1.0 + (std::f64::consts::PI * prog.min(1.0)).cos())
        };
        opt.set_learning_rate(lr);

        let (x, y) = tirer_lot(&sources, false, p.batch.max(1), long, &mut rng)?;
        let (b, t) = x.dims2()?;
        let (perte, norme_grad) = {
            let m = verrou(moteur);
            let logits = m.modele.forward(&x)?.reshape((b * t, VOCAB))?.to_dtype(DType::F32)?;
            let perte = loss::cross_entropy(&logits, &y)?;
            let valeur = if pas % 50 == 0 {
                let valeur = perte.to_scalar::<f32>()?;
                if !valeur.is_finite() {
                    return Err(format!("NaN/Inf détecté dans la loss d'entraînement au pas {pas}").into());
                }
                Some(valeur)
            } else { None };
            let mut grads = perte.backward()?;
            let norme = clip_grad_norm(&mut grads, &vars, NORME_GRAD_MAX)?;
            opt.step(&grads)?;
            (valeur, norme)
        };

        let mut val_perte = None;
        if pas == 1 || pas % EVAL_PERIODIQUE == 0 || pas == p.pas {
            let mut moyenne = 0.0f64;
            let nb_eval = 4;
            for _ in 0..nb_eval {
                moyenne += evaluer_loss(moteur, &sources, p.batch.min(8).max(1), long, &mut rng)? as f64;
            }
            let v = (moyenne / nb_eval as f64) as f32;
            if !v.is_finite() {
                return Err(format!("NaN/Inf dans la loss de validation au pas {pas}").into());
            }
            val_perte = Some(v);
            if v < meilleure_val {
                meilleure_val = v;
                verrou(moteur).sauver(FICHIER_POIDS)?;
                verrou(etat).log.push(format!("Nouveau meilleur modèle (validation {v:.4}) sauvegardé dans {FICHIER_POIDS}."));
            }
        }

        {
            let mut e = verrou(etat);
            e.pas = pas;
            if let Some(perte) = perte {
                e.pertes.push((pas, perte));
            }
            if let Some(v) = val_perte {
                e.val_pertes.push((pas, v));
            }
            if pas % 50 == 0 || val_perte.is_some() {
                let affichage_perte = perte.map(|v| format!("{v:.4}")).unwrap_or_else(|| "--".into());
                e.log.push(format!("pas {pas}/{} - perte {affichage_perte} - validation {} - lr {lr:.2e} - norme grad {norme_grad:.3}", p.pas,
                    val_perte.map(|v| format!("{v:.4}")).unwrap_or_else(|| "--".into())));
            }
        }
        if pas % SAUVEGARDE_PERIODIQUE == 0 {
            verrou(moteur).sauver(FICHIER_DERNIER)?;
            verrou(etat).log.push(format!("Checkpoint last sauvegardé (pas {pas})."));
        }
    }
    verrou(moteur).sauver(FICHIER_DERNIER)?;
    verrou(etat).log.push(format!("Checkpoint last sauvegardé dans {FICHIER_DERNIER}. Le meilleur modèle de validation est dans {FICHIER_POIDS} (loss {meilleure_val:.4})."));
    Ok(())
}

// =====================================================================
//  INTERFACE
// =====================================================================
#[derive(PartialEq, Clone, Copy)]
enum Onglet {
    Entrainement,
    Discussion,
    Creation,
}

#[derive(PartialEq, Clone, Copy)]
enum ModeCreation {
    Continuer,
    Description,
}

struct App {
    moteur: Arc<Mutex<Moteur>>,
    onglet: Onglet,
    nb_params: usize,
    device_name: String,
    message: String,

    // Entraînement
    etat: Arc<Mutex<EtatTrain>>,
    chemin_donnees: String,
    pas: usize,
    batch: usize,
    longueur: usize,
    lr: f64,

    // Réglages de génération (communs)
    temperature: f64,
    top_k: usize,

    // Discussion
    historique: Vec<(bool, String)>, // (est_utilisateur, texte)
    saisie_chat: String,
    gen_chat: Arc<Mutex<Gen>>,

    // Création
    mode_creation: ModeCreation,
    saisie_creation: String,
    prefixe_creation: String,
    longueur_creation: usize,
    chemin_export: String,
    gen_creation: Arc<Mutex<Gen>>,
}

impl App {
    fn new(moteur: Arc<Mutex<Moteur>>, message: String, nb_params: usize, device_name: String) -> Self {
        Self {
            moteur,
            onglet: Onglet::Entrainement,
            nb_params,
            device_name,
            message,
            etat: Arc::new(Mutex::new(EtatTrain::default())),
            chemin_donnees: "donnees.txt;donnees_chat.txt".into(),
            pas: 3000,
            batch: BATCH,
            longueur: CONTEXTE,
            lr: LR_MAX,
            temperature: 0.5,
            top_k: 40,
            historique: Vec::new(),
            saisie_chat: String::new(),
            gen_chat: Arc::new(Mutex::new(Gen::default())),
            mode_creation: ModeCreation::Continuer,
            saisie_creation: "def ".into(),
            prefixe_creation: String::new(),
            longueur_creation: 400,
            chemin_export: "creation.py".into(),
            gen_creation: Arc::new(Mutex::new(Gen::default())),
        }
    }

    fn lancer_entrainement(&mut self) {
        {
            let mut e = verrou(&self.etat);
            if e.en_cours {
                return;
            }
            *e = EtatTrain::default();
            e.en_cours = true;
            e.total = self.pas;
            e.debut = Some(Instant::now());
        }
        let moteur = self.moteur.clone();
        let etat = self.etat.clone();
        let p = ParamsTrain {
            chemin: self.chemin_donnees.clone(),
            pas: self.pas,
            batch: self.batch,
            longueur: self.longueur,
            lr: self.lr,
        };
        std::thread::spawn(move || {
            let r = entrainement(&moteur, &etat, &p);
            let mut e = verrou(&etat);
            if let Err(err) = r {
                e.log.push(format!("ERREUR : {err}"));
            }
            e.en_cours = false;
        });
    }

    fn recuperer_chat(&mut self) {
        let mut g = verrou(&self.gen_chat);
        if g.fini_non_lu {
            g.fini_non_lu = false;
            let t = if g.texte.is_empty() { "(réponse vide)".to_string() } else { g.texte.clone() };
            drop(g);
            self.historique.push((false, t));
        }
    }

    fn envoyer_chat(&mut self) {
        let q = self.saisie_chat.trim().to_string();
        if q.is_empty() {
            return;
        }
        self.historique.push((true, q));
        self.saisie_chat.clear();

        let mut p = String::new();
        for (user, msg) in &self.historique {
            if *user {
                p.push_str(&format!("<|user|>\n{msg}\n"));
            } else {
                p.push_str(&format!("<|assistant|>\n{msg}\n<|end|>\n"));
            }
        }
        p.push_str("<|assistant|>\n");
        lancer_generation(
            self.moteur.clone(),
            self.gen_chat.clone(),
            p,
            400,
            self.temperature,
            self.top_k,
            true,
        );
    }

    fn lancer_creation(&mut self) {
        let (prompt, chat) = match self.mode_creation {
            ModeCreation::Continuer => {
                self.prefixe_creation = self.saisie_creation.clone();
                (self.saisie_creation.clone(), false)
            }
            ModeCreation::Description => {
                self.prefixe_creation = String::new();
                (
                    format!("<|user|>\n{}\n<|assistant|>\n", self.saisie_creation.trim()),
                    true,
                )
            }
        };
        lancer_generation(
            self.moteur.clone(),
            self.gen_creation.clone(),
            prompt,
            self.longueur_creation,
            self.temperature,
            self.top_k,
            chat,
        );
    }

    fn ui_reglages(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("Température :");
            ui.add(egui::Slider::new(&mut self.temperature, 0.1..=1.5));
            ui.label("Top-k :");
            ui.add(egui::Slider::new(&mut self.top_k, 1..=VOCAB));
        });
        ui.small("Température basse (0.2-0.4) = code plus strict, haute (0.8+) = plus créatif.");
    }

    // ---------------- Onglet Entraînement ----------------
    fn ui_entrainement(&mut self, ui: &mut egui::Ui) {
        let (en_cours, pas, total, debut) = {
            let e = verrou(&self.etat);
            (e.en_cours, e.pas, e.total, e.debut)
        };

        ui.heading("Entraînement du modèle");
        ui.small("Le modèle apprend à prédire l'octet suivant dans ton fichier de données.");
        ui.add_space(6.0);

        ui.horizontal(|ui| {
            ui.label("Fichiers de données (séparés par ;) :");
            ui.add_enabled(
                !en_cours,
                egui::TextEdit::singleline(&mut self.chemin_donnees).desired_width(340.0),
            );
            let mut total = 0u64;
            let mut manquant: Option<String> = None;
            for c in chemins(&self.chemin_donnees) {
                match std::fs::metadata(&c) {
                    Ok(m) => total += m.len(),
                    Err(_) => manquant = Some(c),
                }
            }
            match manquant {
                Some(c) => ui.colored_label(egui::Color32::LIGHT_RED, format!("introuvable : {c}")),
                None => ui.label(format!("{:.1} Mo au total", total as f64 / 1e6)),
            };
        });

        ui.add_enabled_ui(!en_cours, |ui| {
            egui::Grid::new("params").num_columns(2).show(ui, |ui| {
                ui.label("Nombre de pas :");
                ui.add(egui::DragValue::new(&mut self.pas).range(10..=500_000).speed(50));
                ui.end_row();
                ui.label("Taille de lot (batch) :");
                ui.add(egui::DragValue::new(&mut self.batch).range(1..=128));
                ui.end_row();
                ui.label("Longueur de séquence :");
                ui.add(egui::DragValue::new(&mut self.longueur).range(16..=CONTEXTE));
                ui.end_row();
                ui.label("Learning rate :");
                ui.add(egui::Slider::new(&mut self.lr, 1e-5..=LR_MAX).logarithmic(true));
                ui.end_row();
            });
        });

        ui.add_space(6.0);
        ui.horizontal(|ui| {
            if ui.add_enabled(!en_cours, egui::Button::new("Lancer l'entraînement")).clicked() {
                self.lancer_entrainement();
            }
            if ui.add_enabled(en_cours, egui::Button::new("Arrêter (et sauvegarder)")).clicked() {
                verrou(&self.etat).stop = true;
            }
            ui.separator();
            if ui.add_enabled(!en_cours, egui::Button::new("Sauver les poids")).clicked() {
                self.message = match verrou(&self.moteur).sauver(FICHIER_POIDS) {
                    Ok(_) => format!("Poids sauvegardés dans {FICHIER_POIDS}."),
                    Err(e) => format!("Erreur de sauvegarde : {e}"),
                };
            }
            if ui.add_enabled(!en_cours, egui::Button::new("Charger les poids")).clicked() {
                self.message = match verrou(&self.moteur).charger(FICHIER_POIDS) {
                    Ok(_) => format!("Poids chargés depuis {FICHIER_POIDS}."),
                    Err(e) => format!("Chargement impossible : {e}"),
                };
            }
            if ui.add_enabled(!en_cours, egui::Button::new("Réinitialiser le modèle")).clicked() {
                self.message = match Moteur::new() {
                    Ok(m) => {
                        *verrou(&self.moteur) = m;
                        *verrou(&self.etat) = EtatTrain::default();
                        "Modèle réinitialisé (poids aléatoires).".into()
                    }
                    Err(e) => format!("Erreur : {e}"),
                };
            }
        });
        if !self.message.is_empty() {
            ui.small(&self.message);
        }

        ui.separator();
        if total > 0 {
            let frac = pas as f32 / total as f32;
            let mut txt = format!("{pas} / {total}");
            if let (Some(d), true) = (debut, pas > 0) {
                let ecoule = d.elapsed().as_secs_f64();
                let reste = ecoule / pas as f64 * (total - pas) as f64;
                txt.push_str(&format!(
                    "  -  {:.2} pas/s  -  reste ~{}",
                    pas as f64 / ecoule,
                    fmt_duree(if en_cours { reste } else { 0.0 })
                ));
            }
            ui.add(egui::ProgressBar::new(frac).text(txt));
        }

        let (pertes, val) = {
            let e = verrou(&self.etat);
            (e.pertes.clone(), e.val_pertes.clone())
        };
        courbe(ui, &pertes, &val);

        ui.label("Journal :");
        egui::ScrollArea::vertical()
            .id_salt("journal")
            .max_height(ui.available_height().max(80.0))
            .stick_to_bottom(true)
            .show(ui, |ui| {
                let e = verrou(&self.etat);
                for l in e.log.iter().rev().take(60).collect::<Vec<_>>().into_iter().rev() {
                    ui.monospace(l);
                }
            });
    }

    // ---------------- Onglet Discussion ----------------
    fn ui_saisie_chat(&mut self, ui: &mut egui::Ui) {
        let occupe = verrou(&self.gen_chat).en_cours;
        let id = egui::Id::new("saisie_chat");
        let mut envoyer = false;
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            let focus = ui.memory(|m| m.has_focus(id));
            if focus
                && !occupe
                && ui.input(|i| i.key_pressed(egui::Key::Enter) && !i.modifiers.shift)
            {
                ui.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Enter));
                envoyer = true;
            }
            ui.add_sized(
                [ui.available_width() - 100.0, 60.0],
                egui::TextEdit::multiline(&mut self.saisie_chat)
                    .id(id)
                    .hint_text("Pose ta question (Entrée = envoyer, Maj+Entrée = nouvelle ligne)"),
            );
            ui.vertical(|ui| {
                if ui.add_enabled(!occupe, egui::Button::new("Envoyer")).clicked() {
                    envoyer = true;
                }
                if ui.add_enabled(occupe, egui::Button::new("Stop")).clicked() {
                    verrou(&self.gen_chat).annuler = true;
                }
            });
        });
        ui.add_space(6.0);
        if envoyer {
            self.envoyer_chat();
            ui.memory_mut(|m| m.request_focus(id));
        }
    }

    fn ui_chat(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.heading("Discussion");
            if ui.button("Nouvelle discussion").clicked() {
                self.historique.clear();
            }
        });
        self.ui_reglages(ui);
        ui.separator();

        let (live, texte_live) = {
            let g = verrou(&self.gen_chat);
            (g.en_cours, g.texte.clone())
        };
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .stick_to_bottom(true)
            .show(ui, |ui| {
                if self.historique.is_empty() && !live {
                    ui.weak("Aucun message. Le modèle doit avoir été entraîné sur des données au format <|user|> ... <|assistant|> ... <|end|>.");
                }
                for (user, msg) in &self.historique {
                    afficher_message(ui, *user, msg);
                }
                if live {
                    afficher_message(ui, false, &format!("{texte_live} ▌"));
                }
            });
    }

    // ---------------- Onglet Création ----------------
    fn ui_creation(&mut self, ui: &mut egui::Ui) {
        ui.heading("Création de code");
        self.ui_reglages(ui);
        ui.horizontal(|ui| {
            ui.radio_value(&mut self.mode_creation, ModeCreation::Continuer, "Continuer un code (début du code)");
            ui.radio_value(&mut self.mode_creation, ModeCreation::Description, "Générer depuis une description");
        });
        ui.add(
            egui::TextEdit::multiline(&mut self.saisie_creation)
                .code_editor()
                .desired_rows(6)
                .desired_width(f32::INFINITY)
                .hint_text(match self.mode_creation {
                    ModeCreation::Continuer => "Début du code à continuer, ex : def tri_rapide(liste):",
                    ModeCreation::Description => "Décris la fonction à créer, ex : fonction qui calcule la factorielle",
                }),
        );

        let occupe = verrou(&self.gen_creation).en_cours;
        ui.horizontal(|ui| {
            ui.label("Longueur max (octets) :");
            ui.add(egui::Slider::new(&mut self.longueur_creation, 50..=2000));
            if ui.add_enabled(!occupe, egui::Button::new("Générer")).clicked() {
                self.lancer_creation();
            }
            if ui.add_enabled(occupe, egui::Button::new("Stop")).clicked() {
                verrou(&self.gen_creation).annuler = true;
            }
        });

        let texte = verrou(&self.gen_creation).texte.clone();
        let complet = format!("{}{}", self.prefixe_creation, texte);

        ui.horizontal(|ui| {
            if ui.button("Copier").clicked() {
                ui.ctx().copy_text(complet.clone());
            }
            ui.label("Enregistrer dans :");
            ui.add(egui::TextEdit::singleline(&mut self.chemin_export).desired_width(160.0));
            if ui.button("Enregistrer").clicked() {
                self.message = match std::fs::write(&self.chemin_export, &complet) {
                    Ok(_) => format!("Code enregistré dans {}.", self.chemin_export),
                    Err(e) => format!("Erreur : {e}"),
                };
            }
        });
        if !self.message.is_empty() {
            ui.small(&self.message);
        }
        ui.separator();

        egui::ScrollArea::vertical()
            .id_salt("sortie_creation")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                let suite = if occupe { format!("{complet}▌") } else { complet };
                ui.add(egui::Label::new(egui::RichText::new(suite).monospace()).selectable(true));
            });
    }
}

fn afficher_message(ui: &mut egui::Ui, user: bool, msg: &str) {
    ui.group(|ui| {
        ui.set_width(ui.available_width() - 10.0);
        if user {
            ui.colored_label(egui::Color32::from_rgb(100, 180, 255), "Vous :");
            ui.label(msg);
        } else {
            ui.colored_label(egui::Color32::from_rgb(100, 230, 140), "IA :");
            ui.add(egui::Label::new(egui::RichText::new(msg).monospace()).selectable(true));
        }
    });
    ui.add_space(6.0);
}

fn fmt_duree(s: f64) -> String {
    let s = s as u64;
    format!("{}h{:02}m{:02}s", s / 3600, (s % 3600) / 60, s % 60)
}

fn courbe(ui: &mut egui::Ui, pertes: &[(usize, f32)], val: &[(usize, f32)]) {
    let (rect, _) = ui.allocate_exact_size(
        egui::vec2(ui.available_width(), 190.0),
        egui::Sense::hover(),
    );
    let p = ui.painter_at(rect);
    p.rect_filled(rect, 4.0, egui::Color32::from_gray(25));
    if pertes.len() < 2 {
        p.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "Courbe de perte (apparaît pendant l'entraînement)",
            egui::FontId::proportional(13.0),
            egui::Color32::GRAY,
        );
        return;
    }
    let mut vmax = pertes.iter().map(|(_, v)| *v).fold(f32::MIN, f32::max);
    let mut vmin = pertes.iter().map(|(_, v)| *v).fold(f32::MAX, f32::min);
    for (_, v) in val {
        vmax = vmax.max(*v);
        vmin = vmin.min(*v);
    }
    let plage = (vmax - vmin).max(1e-3);
    let dernier_pas = pertes.last().map(|(pas, _)| *pas).unwrap_or(1).max(1) as f32;
    let marge = 8.0;
    let pos = |i: f32, v: f32| {
        egui::pos2(
            rect.left() + marge + (i / dernier_pas) * (rect.width() - 2.0 * marge),
            rect.bottom() - marge - ((v - vmin) / plage) * (rect.height() - 2.0 * marge),
        )
    };
    let pts: Vec<egui::Pos2> = pertes.iter().map(|(pas, v)| pos(*pas as f32, *v)).collect();
    p.add(egui::Shape::line(pts, egui::Stroke::new(1.2_f32, egui::Color32::from_rgb(100, 180, 255))));
    if val.len() >= 2 {
        let pv: Vec<egui::Pos2> = val.iter().map(|(s, v)| pos(*s as f32, *v)).collect();
        p.add(egui::Shape::line(pv, egui::Stroke::new(2.0_f32, egui::Color32::from_rgb(255, 170, 60))));
    }
    let derniere = pertes.last().map(|(_, v)| *v).unwrap_or(0.0);
    let mut legende = format!("perte {derniere:.3} (bleu)");
    if let Some((_, v)) = val.last() {
        legende.push_str(&format!("  |  validation {v:.3} (orange)"));
    }
    p.text(
        rect.left_top() + egui::vec2(10.0, 6.0),
        egui::Align2::LEFT_TOP,
        legende,
        egui::FontId::monospace(12.0),
        egui::Color32::WHITE,
    );
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.recuperer_chat();

        let actif = verrou(&self.etat).en_cours
            || verrou(&self.gen_chat).en_cours
            || verrou(&self.gen_creation).en_cours;
        if actif {
            ctx.request_repaint_after(Duration::from_millis(500));
        }

        egui::TopBottomPanel::top("haut").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.onglet, Onglet::Entrainement, "Entraînement");
                ui.selectable_value(&mut self.onglet, Onglet::Discussion, "Discussion");
                ui.selectable_value(&mut self.onglet, Onglet::Creation, "Création");
                ui.separator();
                ui.label(format!("{:.2} M paramètres", self.nb_params as f64 / 1e6));
                ui.label(format!("Device : {}", self.device_name));
                if verrou(&self.etat).en_cours {
                    ui.colored_label(egui::Color32::LIGHT_GREEN, "entraînement en cours");
                }
            });
        });

        match self.onglet {
            Onglet::Entrainement => {
                egui::CentralPanel::default().show(ctx, |ui| self.ui_entrainement(ui));
            }
            Onglet::Discussion => {
                egui::TopBottomPanel::bottom("saisie_chat_panel")
                    .show(ctx, |ui| self.ui_saisie_chat(ui));
                egui::CentralPanel::default().show(ctx, |ui| self.ui_chat(ui));
            }
            Onglet::Creation => {
                egui::CentralPanel::default().show(ctx, |ui| self.ui_creation(ui));
            }
        }
    }
}

fn main() -> Res<()> {
    let mut moteur = Moteur::new()?;
    let message = if Path::new(FICHIER_POIDS).exists() {
        match moteur.charger(FICHIER_POIDS) {
            Ok(_) => format!("Poids chargés depuis {FICHIER_POIDS}."),
            Err(e) => format!(
                "Poids incompatibles ({e}). Supprime {FICHIER_POIDS} puis relance un entraînement."
            ),
        }
    } else {
        format!("Aucun fichier {FICHIER_POIDS} : le modèle est vierge, lance un entraînement.")
    };
    let nb_params = moteur.nb_params();
    let device_name = format!("{:?}", moteur.device);
    let moteur = Arc::new(Mutex::new(moteur));

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1000.0, 720.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Studio IA Code - Rust",
        options,
        Box::new(move |_cc| Ok(Box::new(App::new(moteur, message, nb_params, device_name)))),
    )?;
    Ok(())
}