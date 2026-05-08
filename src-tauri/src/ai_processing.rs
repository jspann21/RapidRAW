use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::time::Duration;

use anyhow::Result;
use image::imageops::{self, FilterType};
use image::{
    DynamicImage, GenericImageView, GrayImage, ImageBuffer, Luma, Rgb, Rgb32FImage, Rgba, RgbaImage,
};
use ndarray::{Array, Array4, IxDyn};
use ort::session::Session;
use ort::value::Tensor;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tauri::Emitter;
use tauri::Manager;
use tokenizers::Tokenizer;
use tokio::sync::Mutex as TokioMutex;

const ENCODER_URL: &str = "https://huggingface.co/CyberTimon/RapidRAW-Models/resolve/main/sam_vit_b_01ec64_encoder.onnx?download=true";
const DECODER_URL: &str = "https://huggingface.co/CyberTimon/RapidRAW-Models/resolve/main/sam_vit_b_01ec64_decoder.onnx?download=true";
const ENCODER_FILENAME: &str = "sam_vit_b_01ec64_encoder.onnx";
const DECODER_FILENAME: &str = "sam_vit_b_01ec64_decoder.onnx";
const SAM_INPUT_SIZE: u32 = 1024;
const ENCODER_SHA256: &str = "16ab73d9c824886f0de2938c19df22fb9ec3deebfd0de58e65177e479213d7d1";
const DECODER_SHA256: &str = "85d0d672cf5b7fe763edcde429e5533e62f674af4b15c7d688b7673b0ef00bf7";

const U2NETP_URL: &str =
    "https://huggingface.co/CyberTimon/RapidRAW-Models/resolve/main/u2net.onnx?download=true";
const U2NETP_FILENAME: &str = "u2net.onnx";
const U2NETP_INPUT_SIZE: u32 = 320;
const U2NETP_SHA256: &str = "8d10d2f3bb75ae3b6d527c77944fc5e7dcd94b29809d47a739a7a728a912b491";

const SKYSEG_URL: &str = "https://huggingface.co/CyberTimon/RapidRAW-Models/resolve/main/skyseg-u2net.onnx?download=true";
const SKYSEG_FILENAME: &str = "skyseg_u2net.onnx";
const SKYSEG_INPUT_SIZE: u32 = 320;
const SKYSEG_SHA256: &str = "ab9c34c64c3d821220a2886a4a06da4642ffa14d5b30e8d5339056a089aa1d39";

const CLIP_MODEL_URL: &str =
    "https://huggingface.co/CyberTimon/RapidRAW-Models/resolve/main/clip_model.onnx?download=true";
const CLIP_MODEL_FILENAME: &str = "clip_model.onnx";
const CLIP_TOKENIZER_URL: &str = "https://huggingface.co/CyberTimon/RapidRAW-Models/resolve/main/clip_tokenizer.json?download=true";
const CLIP_TOKENIZER_FILENAME: &str = "clip_tokenizer.json";
const CLIP_MODEL_SHA256: &str = "57879bb1c23cdeb350d23569dd251ed4b740a96d747c529e94a2bb8040ac5d00";

const DENOISE_URL: &str = "https://huggingface.co/CyberTimon/RapidRAW-Models/resolve/main/nind_denoise_utnet_684.onnx?download=true";
const DENOISE_FILENAME: &str = "nind_denoise_utnet_684.onnx";
const DENOISE_SHA256: &str = "ee3586279d514df557ff3f7dec6df37fafc51ba5d3a3435b2cc9ac2d9017e7fe";

pub const LAMA_URL: &str =
    "https://huggingface.co/CyberTimon/RapidRAW-Models/resolve/main/lama_fp16.onnx?download=true";
pub const LAMA_FILENAME: &str = "lama_fp16.onnx";
pub const LAMA_SHA256: &str = "2d6be6277c400d6f1b91819737f7c3da935e5c63d1b521d393be1196a2bfa82c";
const CUDA_DOWNLOAD_URL: &str = "https://developer.nvidia.com/cuda-downloads";
const CUDNN_DOWNLOAD_URL: &str = "https://developer.nvidia.com/cudnn-downloads";
const CUDNN_WINDOWS_INSTALL_GUIDE_URL: &str =
    "https://docs.nvidia.com/deeplearning/cudnn/installation/latest/windows.html";
const NVIDIA_SMI_TIMEOUT: Duration = Duration::from_millis(1500);

const DEPTH_URL: &str = "https://huggingface.co/CyberTimon/RapidRAW-Models/resolve/main/depth_anything_v2_vits.onnx?download=true";
const DEPTH_FILENAME: &str = "depth_anything_v2_vits.onnx";
const DEPTH_INPUT_SIZE: u32 = 518;
const DEPTH_SHA256: &str = "d2b11a11c1d4a12b47608fa65a17ee9a4c605b55ee1730c8e3b526304f2562be";

pub struct AiModels {
    pub sam_encoder: Mutex<Session>,
    pub sam_decoder: Mutex<Session>,
    pub u2netp: Mutex<Session>,
    pub sky_seg: Mutex<Session>,
    pub depth_anything: Mutex<Session>,
}

pub struct ClipModels {
    pub model: Mutex<Session>,
    pub tokenizer: Tokenizer,
}

#[derive(Clone)]
pub struct ImageEmbeddings {
    pub path_hash: String,
    pub embeddings: Array<f32, IxDyn>,
    pub original_size: (u32, u32),
}

#[derive(Clone)]
pub struct CachedDepthMap {
    pub path_hash: String,
    pub depth_image: GrayImage,
    pub original_size: (u32, u32),
}

pub struct AiState {
    pub models: Option<Arc<AiModels>>,
    pub denoise_model: Option<Arc<Mutex<Session>>>,
    pub clip_models: Option<Arc<ClipModels>>,
    pub lama_model: Option<Arc<Mutex<Session>>>,
    pub lama_cuda_model: Option<Arc<Mutex<Session>>>,
    pub embeddings: Option<ImageEmbeddings>,
    pub depth_map: Option<CachedDepthMap>,
}

fn edt_1d(f: &mut [f32], v: &mut [usize], z: &mut [f32], d: &mut [f32]) {
    let n = f.len();
    if n == 0 {
        return;
    }
    let mut k = 0;
    v[0] = 0;
    z[0] = f32::NEG_INFINITY;
    z[1] = f32::INFINITY;
    for q in 1..n {
        let mut s = ((f[q] + (q * q) as f32) - (f[v[k]] + (v[k] * v[k]) as f32))
            / (2.0 * (q as f32 - v[k] as f32));
        while s <= z[k] {
            if k == 0 {
                break;
            }
            k -= 1;
            s = ((f[q] + (q * q) as f32) - (f[v[k]] + (v[k] * v[k]) as f32))
                / (2.0 * (q as f32 - v[k] as f32));
        }
        k += 1;
        v[k] = q;
        z[k] = s;
        z[k + 1] = f32::INFINITY;
    }
    k = 0;
    for (q, d_q) in d[..n].iter_mut().enumerate() {
        while z[k + 1] < q as f32 {
            k += 1;
        }
        let diff = q as f32 - v[k] as f32;
        *d_q = diff * diff + f[v[k]];
    }
    f.copy_from_slice(&d[..n]);
}

fn edt_2d(grid: &[bool], width: usize, height: usize) -> Vec<f32> {
    let area = width * height;
    let mut f = vec![0.0; area];
    for i in 0..area {
        f[i] = if grid[i] { 1e10 } else { 0.0 };
    }

    let max_dim = width.max(height);
    let mut v = vec![0; max_dim];
    let mut z = vec![0.0; max_dim + 1];
    let mut d = vec![0.0; max_dim];

    for y in 0..height {
        let start = y * width;
        let end = start + width;
        edt_1d(&mut f[start..end], &mut v, &mut z, &mut d);
    }

    let mut col = vec![0.0; height];
    for x in 0..width {
        for y in 0..height {
            col[y] = f[y * width + x];
        }
        edt_1d(&mut col, &mut v, &mut z, &mut d);
        for y in 0..height {
            f[y * width + x] = col[y];
        }
    }

    f.into_iter().map(|v| v.sqrt()).collect()
}

pub fn get_models_dir(_app_handle: &tauri::AppHandle) -> Result<PathBuf> {
    let exe_path = std::env::current_exe()?;
    let install_dir = exe_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("Could not resolve RapidRAW executable directory"))?;
    let models_dir = install_dir.join("models");
    if !models_dir.exists() {
        fs::create_dir_all(&models_dir).map_err(|e| {
            anyhow::anyhow!(
                "Could not create install-folder model directory at {}: {}",
                models_dir.display(),
                e
            )
        })?;
    }
    Ok(models_dir)
}

async fn download_model(
    app_handle: Option<&tauri::AppHandle>,
    model_name: Option<&str>,
    url: &str,
    dest: &Path,
) -> Result<()> {
    let mut response = reqwest::get(url).await?.error_for_status()?;
    let total_bytes = response.content_length();
    let mut file = fs::File::create(dest)?;
    let mut downloaded_bytes = 0_u64;
    let mut last_emitted_bytes = 0_u64;
    let emit_step = total_bytes
        .map(|total| (total / 100).max(512 * 1024))
        .unwrap_or(1024 * 1024);

    if let (Some(app_handle), Some(model_name)) = (app_handle, model_name) {
        let _ = app_handle.emit(
            "ai-model-download-progress",
            serde_json::json!({
                "modelName": model_name,
                "downloadedBytes": 0_u64,
                "totalBytes": total_bytes,
            }),
        );
    }

    while let Some(chunk) = response.chunk().await? {
        file.write_all(&chunk)?;
        downloaded_bytes += chunk.len() as u64;

        if downloaded_bytes.saturating_sub(last_emitted_bytes) >= emit_step
            || total_bytes.is_some_and(|total| downloaded_bytes >= total)
        {
            if let (Some(app_handle), Some(model_name)) = (app_handle, model_name) {
                let _ = app_handle.emit(
                    "ai-model-download-progress",
                    serde_json::json!({
                        "modelName": model_name,
                        "downloadedBytes": downloaded_bytes,
                        "totalBytes": total_bytes,
                    }),
                );
            }
            last_emitted_bytes = downloaded_bytes;
        }
    }

    file.flush()?;
    Ok(())
}

fn verify_sha256(path: &Path, expected_hash: &str) -> Result<bool> {
    if !path.exists() {
        return Ok(false);
    }
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    io::copy(&mut file, &mut hasher)?;
    let hash = hasher.finalize();
    let hex_hash = format!("{:x}", hash);
    Ok(hex_hash == expected_hash)
}

#[derive(Serialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct LocalAiGpuInfo {
    pub name: Option<String>,
    pub driver_version: Option<String>,
    pub vram_mb: Option<u64>,
    pub compute_capability: Option<String>,
    pub is_nvidia: bool,
}

#[derive(Serialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct LocalAiModelInfo {
    pub id: String,
    pub name: String,
    pub filename: String,
    pub file_type: String,
    pub required: bool,
    pub installed: bool,
    pub valid: bool,
    pub size_bytes: u64,
    pub sha256: String,
}

#[derive(Serialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct LocalAiRuntimeDependency {
    pub name: String,
    pub kind: String,
    pub found: bool,
    pub path: Option<String>,
}

#[derive(Serialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct LocalAiStatus {
    pub is_windows: bool,
    pub cuda_available: bool,
    pub cuda_provider_available: bool,
    pub cuda_provider_error: Option<String>,
    pub model_dir: String,
    pub model_dir_writable: bool,
    pub model_dir_error: Option<String>,
    pub disk_usage_bytes: u64,
    pub required_file_types: Vec<String>,
    pub runtime_dependencies: Vec<LocalAiRuntimeDependency>,
    pub missing_runtime_dependencies: Vec<String>,
    pub gpu: LocalAiGpuInfo,
    pub models: Vec<LocalAiModelInfo>,
}

struct LocalAiModelSpec {
    id: &'static str,
    name: &'static str,
    filename: &'static str,
    url: &'static str,
    sha256: &'static str,
    file_type: &'static str,
    required: bool,
}

const LOCAL_AI_MODEL_SPECS: &[LocalAiModelSpec] = &[LocalAiModelSpec {
    id: "lama-inpainting",
    name: "LaMa Inpainting",
    filename: LAMA_FILENAME,
    url: LAMA_URL,
    sha256: LAMA_SHA256,
    file_type: ".onnx",
    required: true,
}];

fn find_local_ai_model_spec(model_id: &str) -> Result<&'static LocalAiModelSpec> {
    LOCAL_AI_MODEL_SPECS
        .iter()
        .find(|spec| spec.id == model_id)
        .ok_or_else(|| anyhow::anyhow!("Unknown local AI model: {}", model_id))
}

fn default_gpu_info() -> LocalAiGpuInfo {
    LocalAiGpuInfo {
        name: None,
        driver_version: None,
        vram_mb: None,
        compute_capability: None,
        is_nvidia: false,
    }
}

fn query_nvidia_gpu() -> LocalAiGpuInfo {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let output = Command::new("nvidia-smi")
            .args([
                "--query-gpu=name,driver_version,memory.total,compute_cap",
                "--format=csv,noheader,nounits",
            ])
            .output();
        let _ = tx.send(output);
    });

    let Ok(output) = rx.recv_timeout(NVIDIA_SMI_TIMEOUT) else {
        return default_gpu_info();
    };

    let Ok(output) = output else {
        return default_gpu_info();
    };

    if !output.status.success() {
        return default_gpu_info();
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let first_line = stdout.lines().next().unwrap_or_default();
    let mut parts = first_line.split(',').map(|part| part.trim());
    let name = parts
        .next()
        .filter(|value| !value.is_empty())
        .map(str::to_string);
    let driver_version = parts
        .next()
        .filter(|value| !value.is_empty())
        .map(str::to_string);
    let vram_mb = parts
        .next()
        .and_then(|value| value.split_whitespace().next())
        .and_then(|value| value.parse::<u64>().ok());
    let compute_capability = parts
        .next()
        .filter(|value| !value.is_empty())
        .map(str::to_string);
    let is_nvidia = name
        .as_deref()
        .map(|value| value.to_ascii_lowercase().contains("nvidia"))
        .unwrap_or(false);

    LocalAiGpuInfo {
        name,
        driver_version,
        vram_mb,
        compute_capability,
        is_nvidia,
    }
}

fn check_model_dir_writable(models_dir: &Path) -> (bool, Option<String>) {
    if let Err(e) = fs::create_dir_all(models_dir) {
        return (false, Some(e.to_string()));
    }

    let test_path = models_dir.join(".rapidraw-write-test");
    match fs::write(&test_path, b"ok").and_then(|_| fs::remove_file(&test_path)) {
        Ok(()) => (true, None),
        Err(e) => (false, Some(e.to_string())),
    }
}

fn model_info_for_spec(
    models_dir: &Path,
    spec: &LocalAiModelSpec,
    verify_hash: bool,
) -> Result<LocalAiModelInfo> {
    let path = models_dir.join(spec.filename);
    let installed = path.exists();
    let size_bytes = if installed {
        fs::metadata(&path).map(|m| m.len()).unwrap_or(0)
    } else {
        0
    };
    let valid = if installed && verify_hash {
        verify_sha256(&path, spec.sha256).unwrap_or(false)
    } else {
        installed
    };

    Ok(LocalAiModelInfo {
        id: spec.id.to_string(),
        name: spec.name.to_string(),
        filename: spec.filename.to_string(),
        file_type: spec.file_type.to_string(),
        required: spec.required,
        installed,
        valid,
        size_bytes,
        sha256: spec.sha256.to_string(),
    })
}

fn model_dir_disk_usage(models_dir: &Path) -> u64 {
    fs::read_dir(models_dir)
        .ok()
        .into_iter()
        .flat_map(|entries| entries.flatten())
        .filter_map(|entry| entry.metadata().ok())
        .filter(|metadata| metadata.is_file())
        .map(|metadata| metadata.len())
        .sum()
}

fn path_dir_if_exists(path: impl AsRef<Path>) -> Option<PathBuf> {
    let path = path.as_ref();
    if path.as_os_str().is_empty() {
        return None;
    }
    let dir = if path.is_file() {
        path.parent()?.to_path_buf()
    } else {
        path.to_path_buf()
    };
    if dir.exists() && dir.is_dir() {
        Some(dir)
    } else {
        None
    }
}

fn add_env_path_dirs(dirs: &mut Vec<PathBuf>, env_name: &str) {
    let Some(value) = std::env::var_os(env_name) else {
        return;
    };
    for dir in std::env::split_paths(&value) {
        if let Some(dir) = path_dir_if_exists(dir) {
            dirs.push(dir);
        }
    }
}

fn add_runtime_dir_candidate(dirs: &mut Vec<PathBuf>, path: impl AsRef<Path>) {
    let path = path.as_ref();
    if path.as_os_str().is_empty() {
        return;
    }
    dirs.push(path.to_path_buf());
    dirs.push(path.join("bin"));
}

fn add_dirs_containing_any(
    dirs: &mut Vec<PathBuf>,
    root: &Path,
    filenames: &[&str],
    max_depth: usize,
) {
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };

    let mut child_dirs = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            child_dirs.push(path);
            continue;
        }

        let Some(filename) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if filenames
            .iter()
            .any(|expected| filename.eq_ignore_ascii_case(expected))
        {
            dirs.push(root.to_path_buf());
        }
    }

    if max_depth == 0 {
        return;
    }

    for child in child_dirs {
        add_dirs_containing_any(dirs, &child, filenames, max_depth - 1);
    }
}

fn add_common_cuda_dirs(dirs: &mut Vec<PathBuf>) {
    #[cfg(target_os = "windows")]
    {
        for env_name in [
            "CUDA_PATH",
            "CUDA_PATH_V12_9",
            "CUDA_PATH_V12_8",
            "CUDA_PATH_V12_7",
            "CUDA_PATH_V12_6",
            "CUDA_PATH_V12_5",
            "CUDA_PATH_V12_4",
            "CUDA_PATH_V12_3",
            "CUDA_PATH_V12_2",
            "CUDA_PATH_V12_1",
            "CUDA_PATH_V12_0",
        ] {
            if let Some(value) = std::env::var_os(env_name) {
                dirs.push(PathBuf::from(value).join("bin"));
            }
        }

        let cuda_root = Path::new("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA");
        if let Ok(entries) = fs::read_dir(cuda_root) {
            let mut versions = entries
                .flatten()
                .map(|entry| entry.path().join("bin"))
                .filter(|path| path.exists())
                .collect::<Vec<_>>();
            versions.sort_by(|a, b| b.cmp(a));
            dirs.extend(versions);
        }

        for root in [
            "C:/Program Files/NVIDIA/CUDNN",
            "C:/Program Files/NVIDIA/cuDNN",
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDNN",
        ] {
            if let Ok(entries) = fs::read_dir(root) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    add_runtime_dir_candidate(dirs, path);
                }
            }
            add_dirs_containing_any(
                dirs,
                Path::new(root),
                &ort::execution_providers::cuda::CUDNN_DYLIBS,
                4,
            );
        }
    }
}

fn dedupe_dirs(dirs: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut result = Vec::new();
    let mut seen = Vec::<String>::new();
    for dir in dirs {
        let Some(dir) = path_dir_if_exists(dir) else {
            continue;
        };
        let canonical = dir.canonicalize().unwrap_or(dir);
        let normalized = canonical.to_string_lossy().to_ascii_lowercase();
        if !seen.iter().any(|existing| existing == &normalized) {
            seen.push(normalized);
            result.push(canonical);
        }
    }
    result
}

fn runtime_dependency_names() -> Vec<(&'static str, &'static str)> {
    let mut deps = Vec::new();
    for name in ort::execution_providers::cuda::CUDA_DYLIBS {
        deps.push((*name, "CUDA"));
    }
    for name in ort::execution_providers::cuda::CUDNN_DYLIBS {
        deps.push((*name, "cuDNN"));
    }
    deps
}

fn local_ai_runtime_dirs(app_handle: &tauri::AppHandle) -> Vec<PathBuf> {
    let mut dirs = Vec::new();

    if let Ok(settings) = crate::app_settings::load_settings(app_handle.clone()) {
        if let Some(path) = settings.local_ai_cuda_runtime_path {
            add_runtime_dir_candidate(&mut dirs, path);
        }
        if let Some(path) = settings.local_ai_cudnn_runtime_path {
            add_runtime_dir_candidate(&mut dirs, path);
        }
    }

    if let Ok(resource_path) = app_handle
        .path()
        .resolve("resources", tauri::path::BaseDirectory::Resource)
    {
        dirs.push(resource_path);
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(parent) = exe.parent()
    {
        dirs.push(parent.to_path_buf());
    }

    add_env_path_dirs(&mut dirs, "PATH");
    add_common_cuda_dirs(&mut dirs);
    dedupe_dirs(dirs)
}

fn inspect_runtime_dependencies(
    app_handle: &tauri::AppHandle,
) -> (Vec<LocalAiRuntimeDependency>, Vec<String>, Vec<PathBuf>) {
    if !cfg!(target_os = "windows") {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    let dirs = local_ai_runtime_dirs(app_handle);
    let mut dependencies = Vec::new();
    let mut missing = Vec::new();
    let mut found_dirs = Vec::new();

    for (name, kind) in runtime_dependency_names() {
        let found_path = dirs
            .iter()
            .map(|dir| dir.join(name))
            .find(|path| path.exists());
        if let Some(path) = found_path {
            if let Some(parent) = path.parent() {
                found_dirs.push(parent.to_path_buf());
            }
            dependencies.push(LocalAiRuntimeDependency {
                name: name.to_string(),
                kind: kind.to_string(),
                found: true,
                path: Some(path.to_string_lossy().to_string()),
            });
        } else {
            missing.push(name.to_string());
            dependencies.push(LocalAiRuntimeDependency {
                name: name.to_string(),
                kind: kind.to_string(),
                found: false,
                path: None,
            });
        }
    }

    (dependencies, missing, dedupe_dirs(found_dirs))
}

fn prepend_runtime_dirs_to_path(dirs: &[PathBuf]) {
    if dirs.is_empty() {
        return;
    }

    let current_path = std::env::var_os("PATH").unwrap_or_default();
    let mut paths = std::env::split_paths(&current_path).collect::<Vec<_>>();
    for dir in dirs.iter().rev() {
        if !paths.iter().any(|path| path == dir) {
            paths.insert(0, dir.clone());
        }
    }
    if let Ok(joined) = std::env::join_paths(paths) {
        unsafe {
            std::env::set_var("PATH", joined);
        }
    }
}

fn prepare_cuda_runtime(
    app_handle: &tauri::AppHandle,
) -> (Vec<LocalAiRuntimeDependency>, Vec<String>) {
    let (dependencies, missing, found_dirs) = inspect_runtime_dependencies(app_handle);
    prepend_runtime_dirs_to_path(&found_dirs);
    (dependencies, missing)
}

fn missing_runtime_dependency_help(missing: &[String]) -> String {
    let has_cuda = missing.iter().any(|name| {
        let lower = name.to_ascii_lowercase();
        lower.contains("cuda") || lower.contains("cublas")
    });
    let has_cudnn = missing
        .iter()
        .any(|name| name.to_ascii_lowercase().contains("cudnn"));
    let mut help = Vec::new();

    match (has_cuda, has_cudnn) {
        (true, true) => help.push(format!(
            "CUDA Toolkit 12.x and cuDNN 9 are missing. Install CUDA from {CUDA_DOWNLOAD_URL}, then install cuDNN from {CUDNN_DOWNLOAD_URL}. Guide: {CUDNN_WINDOWS_INSTALL_GUIDE_URL}."
        )),
        (true, false) => help.push(format!(
            "CUDA Toolkit 12.x is missing. Install it from {CUDA_DOWNLOAD_URL}, then refresh Local GPU status."
        )),
        (false, true) => help.push(format!(
            "cuDNN 9 is missing. Install it from {CUDNN_DOWNLOAD_URL}, then refresh Local GPU status. Guide: {CUDNN_WINDOWS_INSTALL_GUIDE_URL}."
        )),
        (false, false) => help.push("Required CUDA runtime files are missing. Install CUDA Toolkit 12.x and cuDNN 9, then refresh Local GPU status.".to_string()),
    }

    help.join(" ")
}

async fn download_and_verify_model(
    app_handle: &tauri::AppHandle,
    models_dir: &Path,
    filename: &str,
    url: &str,
    expected_hash: &str,
    model_name: &str,
) -> Result<()> {
    let dest_path = models_dir.join(filename);
    let is_valid = verify_sha256(&dest_path, expected_hash)?;

    if !is_valid {
        if dest_path.exists() {
            println!("Model {} has incorrect hash. Re-downloading.", model_name);
            fs::remove_file(&dest_path)?;
        }
        let _ = app_handle.emit("ai-model-download-start", model_name);
        if let Err(err) = download_model(Some(app_handle), Some(model_name), url, &dest_path).await
        {
            let _ = app_handle.emit("ai-model-download-finish", model_name);
            return Err(err);
        }
        let _ = app_handle.emit("ai-model-download-finish", model_name);

        if !verify_sha256(&dest_path, expected_hash)? {
            return Err(anyhow::anyhow!(
                "Failed to verify model {} after download. Hash mismatch.",
                model_name
            ));
        }
    }
    Ok(())
}

fn cuda_execution_provider() -> ort::execution_providers::ExecutionProviderDispatch {
    ort::execution_providers::CUDAExecutionProvider::default()
        .with_device_id(0)
        .with_tf32(true)
        .with_prefer_nhwc(true)
        .with_conv_algorithm_search(
            ort::execution_providers::cuda::CuDNNConvAlgorithmSearch::Heuristic,
        )
        .build()
        .error_on_failure()
}

fn cuda_session_builder() -> Result<ort::session::builder::SessionBuilder> {
    Ok(Session::builder()?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
        .with_execution_providers([cuda_execution_provider()])?)
}

fn probe_cuda_execution_provider(
    app_handle: &tauri::AppHandle,
) -> (
    bool,
    Option<String>,
    Vec<LocalAiRuntimeDependency>,
    Vec<String>,
) {
    if !cfg!(all(target_os = "windows", target_arch = "x86_64")) {
        return (
            false,
            Some("Local GPU is currently supported on Windows x64 only.".to_string()),
            Vec::new(),
            Vec::new(),
        );
    }

    let (runtime_dependencies, missing_runtime_dependencies) = prepare_cuda_runtime(app_handle);
    if !missing_runtime_dependencies.is_empty() {
        let help = missing_runtime_dependency_help(&missing_runtime_dependencies);
        return (
            false,
            Some(help),
            runtime_dependencies,
            missing_runtime_dependencies,
        );
    }

    let _ = ort::init().with_name("AI-CUDA-Probe").commit();
    match cuda_session_builder() {
        Ok(_) => (
            true,
            None,
            runtime_dependencies,
            missing_runtime_dependencies,
        ),
        Err(e) => {
            let message = e.to_string();
            let lower = message.to_ascii_lowercase();
            let help = if lower.contains("cudnn")
                || lower.contains("cudart")
                || lower.contains("cublas")
            {
                " Install CUDA 12.x and cuDNN 9, or add their bin directories to PATH."
            } else {
                ""
            };
            (
                false,
                Some(format!("{}{}", message, help)),
                runtime_dependencies,
                missing_runtime_dependencies,
            )
        }
    }
}

pub fn get_local_ai_status(
    app_handle: &tauri::AppHandle,
    probe_runtime: bool,
) -> Result<LocalAiStatus> {
    let models_dir = get_models_dir(app_handle)?;
    let gpu = query_nvidia_gpu();
    let (model_dir_writable, model_dir_error) = check_model_dir_writable(&models_dir);
    let models = LOCAL_AI_MODEL_SPECS
        .iter()
        .map(|spec| model_info_for_spec(&models_dir, spec, probe_runtime))
        .collect::<Result<Vec<_>>>()?;
    let (
        cuda_provider_available,
        cuda_provider_error,
        runtime_dependencies,
        missing_runtime_dependencies,
    ) = if probe_runtime {
        probe_cuda_execution_provider(app_handle)
    } else {
        let (runtime_dependencies, missing_runtime_dependencies, found_dirs) =
            inspect_runtime_dependencies(app_handle);
        prepend_runtime_dirs_to_path(&found_dirs);
        (
            false,
            if missing_runtime_dependencies.is_empty() {
                None
            } else {
                Some(missing_runtime_dependency_help(
                    &missing_runtime_dependencies,
                ))
            },
            runtime_dependencies,
            missing_runtime_dependencies,
        )
    };

    Ok(LocalAiStatus {
        is_windows: cfg!(target_os = "windows"),
        cuda_available: gpu.is_nvidia,
        cuda_provider_available,
        cuda_provider_error,
        model_dir: models_dir.to_string_lossy().to_string(),
        model_dir_writable,
        model_dir_error,
        disk_usage_bytes: model_dir_disk_usage(&models_dir),
        required_file_types: vec![".onnx".to_string()],
        runtime_dependencies,
        missing_runtime_dependencies,
        gpu,
        models,
    })
}

pub async fn download_local_ai_model(
    app_handle: &tauri::AppHandle,
    model_id: &str,
) -> Result<LocalAiModelInfo> {
    let spec = find_local_ai_model_spec(model_id)?;
    let models_dir = get_models_dir(app_handle)?;
    let (writable, error) = check_model_dir_writable(&models_dir);
    if !writable {
        return Err(anyhow::anyhow!(
            "Install-folder model directory is not writable: {}",
            error.unwrap_or_else(|| "unknown error".to_string())
        ));
    }

    download_and_verify_model(
        app_handle,
        &models_dir,
        spec.filename,
        spec.url,
        spec.sha256,
        spec.name,
    )
    .await?;

    model_info_for_spec(&models_dir, spec, true)
}

pub fn delete_local_ai_model(
    app_handle: &tauri::AppHandle,
    model_id: &str,
) -> Result<LocalAiModelInfo> {
    let spec = find_local_ai_model_spec(model_id)?;
    let models_dir = get_models_dir(app_handle)?;
    let model_path = models_dir.join(spec.filename);
    let canonical_dir = models_dir.canonicalize()?;

    if model_path.exists() {
        let canonical_model = model_path.canonicalize()?;
        if !canonical_model.starts_with(&canonical_dir) {
            return Err(anyhow::anyhow!(
                "Refusing to delete model outside install-folder model directory"
            ));
        }
        fs::remove_file(&canonical_model)?;
    }

    model_info_for_spec(&models_dir, spec, true)
}

pub async fn run_local_ai_self_test(
    app_handle: &tauri::AppHandle,
    ai_state_mutex: &Mutex<Option<AiState>>,
    ai_init_lock: &TokioMutex<()>,
) -> Result<String> {
    let lama_model = get_or_init_lama_cuda_model(app_handle, ai_state_mutex, ai_init_lock).await?;

    let mut image = RgbaImage::new(64, 64);
    for (x, y, pixel) in image.enumerate_pixels_mut() {
        let value = if (x / 8 + y / 8) % 2 == 0 { 220 } else { 180 };
        *pixel = Rgba([value, value, value, 255]);
    }

    let mut mask = GrayImage::new(64, 64);
    for y in 24..40 {
        for x in 24..40 {
            mask.put_pixel(x, y, Luma([255]));
        }
    }

    let source = DynamicImage::ImageRgba8(image);
    let result = run_lama_inpainting(&source, &mask, &lama_model)?;
    if result.dimensions() != (64, 64) {
        return Err(anyhow::anyhow!(
            "Local GPU self-test returned an unexpected image size."
        ));
    }

    Ok("Local GPU CUDA inpainting self-test completed.".to_string())
}

pub async fn get_or_init_ai_models(
    app_handle: &tauri::AppHandle,
    ai_state_mutex: &Mutex<Option<AiState>>,
    ai_init_lock: &TokioMutex<()>,
) -> Result<Arc<AiModels>> {
    if let Some(models) = ai_state_mutex
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|state| state.models.clone())
    {
        return Ok(models);
    }

    let _guard = ai_init_lock.lock().await;

    if let Some(models) = ai_state_mutex
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|state| state.models.clone())
    {
        return Ok(models);
    }

    let models_dir = get_models_dir(app_handle)?;

    download_and_verify_model(
        app_handle,
        &models_dir,
        ENCODER_FILENAME,
        ENCODER_URL,
        ENCODER_SHA256,
        "SAM Encoder",
    )
    .await?;
    download_and_verify_model(
        app_handle,
        &models_dir,
        DECODER_FILENAME,
        DECODER_URL,
        DECODER_SHA256,
        "SAM Decoder",
    )
    .await?;
    download_and_verify_model(
        app_handle,
        &models_dir,
        U2NETP_FILENAME,
        U2NETP_URL,
        U2NETP_SHA256,
        "Foreground Model",
    )
    .await?;
    download_and_verify_model(
        app_handle,
        &models_dir,
        SKYSEG_FILENAME,
        SKYSEG_URL,
        SKYSEG_SHA256,
        "Sky Model",
    )
    .await?;
    download_and_verify_model(
        app_handle,
        &models_dir,
        DEPTH_FILENAME,
        DEPTH_URL,
        DEPTH_SHA256,
        "Depth Model",
    )
    .await?;

    let _ = ort::init().with_name("AI").commit();

    let encoder_path = models_dir.join(ENCODER_FILENAME);
    let decoder_path = models_dir.join(DECODER_FILENAME);
    let u2netp_path = models_dir.join(U2NETP_FILENAME);
    let sky_seg_path = models_dir.join(SKYSEG_FILENAME);
    let depth_path = models_dir.join(DEPTH_FILENAME);

    let sam_encoder = Session::builder()?.commit_from_file(encoder_path)?;
    let sam_decoder = Session::builder()?.commit_from_file(decoder_path)?;
    let u2netp = Session::builder()?.commit_from_file(u2netp_path)?;
    let sky_seg = Session::builder()?.commit_from_file(sky_seg_path)?;
    let depth_anything = Session::builder()?.commit_from_file(depth_path)?;

    crate::register_exit_handler();

    let models = Arc::new(AiModels {
        sam_encoder: Mutex::new(sam_encoder),
        sam_decoder: Mutex::new(sam_decoder),
        u2netp: Mutex::new(u2netp),
        sky_seg: Mutex::new(sky_seg),
        depth_anything: Mutex::new(depth_anything),
    });

    let mut ai_state_lock = ai_state_mutex.lock().unwrap();
    if let Some(state) = ai_state_lock.as_mut() {
        state.models = Some(models.clone());
    } else {
        *ai_state_lock = Some(AiState {
            models: Some(models.clone()),
            denoise_model: None,
            clip_models: None,
            lama_model: None,
            lama_cuda_model: None,
            embeddings: None,
            depth_map: None,
        });
    }

    Ok(models)
}

pub async fn get_or_init_denoise_model(
    app_handle: &tauri::AppHandle,
    ai_state_mutex: &Mutex<Option<AiState>>,
    ai_init_lock: &TokioMutex<()>,
) -> Result<Arc<Mutex<Session>>> {
    if let Some(denoise_model) = ai_state_mutex
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|state| state.denoise_model.clone())
    {
        return Ok(denoise_model);
    }

    let _guard = ai_init_lock.lock().await;

    if let Some(denoise_model) = ai_state_mutex
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|state| state.denoise_model.clone())
    {
        return Ok(denoise_model);
    }

    let models_dir = get_models_dir(app_handle)?;
    download_and_verify_model(
        app_handle,
        &models_dir,
        DENOISE_FILENAME,
        DENOISE_URL,
        DENOISE_SHA256,
        "NIND Denoise Model",
    )
    .await?;

    let _ = ort::init().with_name("AI-Denoise").commit();
    let model_path = models_dir.join(DENOISE_FILENAME);
    let session = Session::builder()?.commit_from_file(model_path)?;
    let denoise_model = Arc::new(Mutex::new(session));

    crate::register_exit_handler();

    let mut ai_state_lock = ai_state_mutex.lock().unwrap();
    if let Some(state) = ai_state_lock.as_mut() {
        state.denoise_model = Some(denoise_model.clone());
    } else {
        *ai_state_lock = Some(AiState {
            models: None,
            denoise_model: Some(denoise_model.clone()),
            clip_models: None,
            lama_model: None,
            lama_cuda_model: None,
            embeddings: None,
            depth_map: None,
        });
    }

    Ok(denoise_model)
}

pub async fn get_or_init_clip_models(
    app_handle: &tauri::AppHandle,
    ai_state_mutex: &Mutex<Option<AiState>>,
    ai_init_lock: &TokioMutex<()>,
) -> Result<Arc<ClipModels>> {
    if let Some(clip_models) = ai_state_mutex
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|state| state.clip_models.clone())
    {
        return Ok(clip_models);
    }

    let _guard = ai_init_lock.lock().await;

    if let Some(clip_models) = ai_state_mutex
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|state| state.clip_models.clone())
    {
        return Ok(clip_models);
    }

    let models_dir = get_models_dir(app_handle)?;

    download_and_verify_model(
        app_handle,
        &models_dir,
        CLIP_MODEL_FILENAME,
        CLIP_MODEL_URL,
        CLIP_MODEL_SHA256,
        "CLIP Model",
    )
    .await?;

    let clip_tokenizer_path = models_dir.join(CLIP_TOKENIZER_FILENAME);
    if !clip_tokenizer_path.exists() {
        let model_name = "CLIP Tokenizer";
        let _ = app_handle.emit("ai-model-download-start", model_name);
        if let Err(err) = download_model(
            Some(app_handle),
            Some(model_name),
            CLIP_TOKENIZER_URL,
            &clip_tokenizer_path,
        )
        .await
        {
            let _ = app_handle.emit("ai-model-download-finish", model_name);
            return Err(err);
        }
        let _ = app_handle.emit("ai-model-download-finish", "CLIP Tokenizer");
    }

    let _ = ort::init().with_name("AI-Tagging").commit();
    let clip_model_path = models_dir.join(CLIP_MODEL_FILENAME);
    let model = Mutex::new(Session::builder()?.commit_from_file(clip_model_path)?);
    let tokenizer =
        Tokenizer::from_file(clip_tokenizer_path).map_err(|e| anyhow::anyhow!(e.to_string()))?;

    crate::register_exit_handler();

    let clip_models = Arc::new(ClipModels { model, tokenizer });

    let mut ai_state_lock = ai_state_mutex.lock().unwrap();
    if let Some(state) = ai_state_lock.as_mut() {
        state.clip_models = Some(clip_models.clone());
    } else {
        *ai_state_lock = Some(AiState {
            models: None,
            denoise_model: None,
            clip_models: Some(clip_models.clone()),
            lama_model: None,
            lama_cuda_model: None,
            embeddings: None,
            depth_map: None,
        });
    }

    Ok(clip_models)
}

pub async fn get_or_init_lama_model(
    app_handle: &tauri::AppHandle,
    ai_state_mutex: &Mutex<Option<AiState>>,
    ai_init_lock: &TokioMutex<()>,
) -> Result<Arc<Mutex<Session>>> {
    if let Some(lama_model) = ai_state_mutex
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|state| state.lama_model.clone())
    {
        return Ok(lama_model);
    }

    let _guard = ai_init_lock.lock().await;

    if let Some(lama_model) = ai_state_mutex
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|state| state.lama_model.clone())
    {
        return Ok(lama_model);
    }

    let models_dir = get_models_dir(app_handle)?;
    download_and_verify_model(
        app_handle,
        &models_dir,
        LAMA_FILENAME,
        LAMA_URL,
        LAMA_SHA256,
        "Inpainting Model",
    )
    .await?;

    let _ = ort::init().with_name("AI-Inpainting").commit();
    let model_path = models_dir.join(LAMA_FILENAME);
    let session = Session::builder()?.commit_from_file(model_path)?;
    let lama_model = Arc::new(Mutex::new(session));

    crate::register_exit_handler();

    let mut ai_state_lock = ai_state_mutex.lock().unwrap();
    if let Some(state) = ai_state_lock.as_mut() {
        state.lama_model = Some(lama_model.clone());
    } else {
        *ai_state_lock = Some(AiState {
            models: None,
            denoise_model: None,
            clip_models: None,
            lama_model: Some(lama_model.clone()),
            lama_cuda_model: None,
            embeddings: None,
            depth_map: None,
        });
    }

    Ok(lama_model)
}

pub async fn get_or_init_lama_cuda_model(
    app_handle: &tauri::AppHandle,
    ai_state_mutex: &Mutex<Option<AiState>>,
    ai_init_lock: &TokioMutex<()>,
) -> Result<Arc<Mutex<Session>>> {
    if !cfg!(all(target_os = "windows", target_arch = "x86_64")) {
        return Err(anyhow::anyhow!(
            "Local GPU is currently supported on Windows x64 only."
        ));
    }

    if let Some(lama_cuda_model) = ai_state_mutex
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|state| state.lama_cuda_model.clone())
    {
        return Ok(lama_cuda_model);
    }

    let _guard = ai_init_lock.lock().await;

    if let Some(lama_cuda_model) = ai_state_mutex
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|state| state.lama_cuda_model.clone())
    {
        return Ok(lama_cuda_model);
    }

    let models_dir = get_models_dir(app_handle)?;
    let (_, missing_runtime_dependencies) = prepare_cuda_runtime(app_handle);
    if !missing_runtime_dependencies.is_empty() {
        let help = missing_runtime_dependency_help(&missing_runtime_dependencies);
        return Err(anyhow::anyhow!(help));
    }

    download_and_verify_model(
        app_handle,
        &models_dir,
        LAMA_FILENAME,
        LAMA_URL,
        LAMA_SHA256,
        "LaMa Inpainting",
    )
    .await?;

    let _ = ort::init().with_name("AI-Inpainting-CUDA").commit();
    let model_path = models_dir.join(LAMA_FILENAME);
    let session = cuda_session_builder()?.commit_from_file(model_path)?;
    let lama_cuda_model = Arc::new(Mutex::new(session));

    crate::register_exit_handler();

    let mut ai_state_lock = ai_state_mutex.lock().unwrap();
    if let Some(state) = ai_state_lock.as_mut() {
        state.lama_cuda_model = Some(lama_cuda_model.clone());
    } else {
        *ai_state_lock = Some(AiState {
            models: None,
            denoise_model: None,
            clip_models: None,
            lama_model: None,
            lama_cuda_model: Some(lama_cuda_model.clone()),
            embeddings: None,
            depth_map: None,
        });
    }

    Ok(lama_cuda_model)
}

#[derive(Clone, Copy)]
struct TileParams {
    cs: usize,
    ucs: usize,
    overlap: usize,
    pad: usize,
}

impl TileParams {
    const fn new(cs: usize, ucs: usize, overlap: usize) -> Self {
        Self {
            cs,
            ucs,
            overlap,
            pad: (cs - ucs) / 2,
        }
    }
}

const TILE_BALANCED: TileParams = TileParams::new(504, 480, 6);
const TILE_FASTER: TileParams = TileParams::new(504, 504, 0);
const TILE_HIGHER_QUALITY: TileParams = TileParams::new(504, 448, 12);

fn select_tile_params(quality_0_1: f32) -> TileParams {
    let q = quality_0_1.clamp(0.0, 1.0);
    if q <= 0.25 {
        TILE_FASTER
    } else if q >= 0.75 {
        TILE_HIGHER_QUALITY
    } else {
        TILE_BALANCED
    }
}

#[inline]
fn mirror_coord(c: i32, size: i32) -> i32 {
    if c < 0 {
        (-c).min(size - 1)
    } else if c >= size {
        (2 * size - 1 - c).max(0)
    } else {
        c
    }
}

fn extract_tile_mirror(img: &Rgb32FImage, x0: i32, y0: i32, cs: usize) -> Array4<f32> {
    let (w, h) = (img.width() as i32, img.height() as i32);
    let mut arr = Array4::zeros((1, 3, cs, cs));
    for dy in 0..cs as i32 {
        for dx in 0..cs as i32 {
            let sx = mirror_coord(x0 + dx, w);
            let sy = mirror_coord(y0 + dy, h);
            let px = img.get_pixel(sx as u32, sy as u32);
            arr[[0, 0, dy as usize, dx as usize]] = px[0];
            arr[[0, 1, dy as usize, dx as usize]] = px[1];
            arr[[0, 2, dy as usize, dx as usize]] = px[2];
        }
    }
    arr
}

struct SeamlessBlend {
    ud0: usize,
    ud1: usize,
    ud2: usize,
    ud3: usize,
    absx0: usize,
    absy0: usize,
    fswidth: usize,
    fsheight: usize,
    overlap: usize,
}

fn apply_seamless(tile: &mut Array4<f32>, blend: &SeamlessBlend) {
    let SeamlessBlend {
        ud0,
        ud1,
        ud2,
        ud3,
        absx0,
        absy0,
        fswidth,
        fsheight,
        overlap,
    } = *blend;
    let ol = overlap;
    if absx0 > 0 {
        for c in 0..3 {
            for y in ud1..ud3 {
                for x in ud0..(ud0 + ol).min(ud2) {
                    tile[[0, c, y, x]] *= 0.5;
                }
            }
        }
    }
    if absy0 > 0 {
        for c in 0..3 {
            for y in ud1..(ud1 + ol).min(ud3) {
                for x in ud0..ud2 {
                    tile[[0, c, y, x]] *= 0.5;
                }
            }
        }
    }
    if absx0 + (ud2 - ud0) < fswidth && ol > 0 {
        let right_start = (ud2 as i32 - ol as i32).max(ud0 as i32) as usize;
        for c in 0..3 {
            for y in ud1..ud3 {
                for x in right_start..ud2 {
                    tile[[0, c, y, x]] *= 0.5;
                }
            }
        }
    }
    if absy0 + (ud3 - ud1) < fsheight && ol > 0 {
        let bottom_start = (ud3 as i32 - ol as i32).max(ud1 as i32) as usize;
        for c in 0..3 {
            for y in bottom_start..ud3 {
                for x in ud0..ud2 {
                    tile[[0, c, y, x]] *= 0.5;
                }
            }
        }
    }
}

fn run_native_denoise(
    img: &Rgb32FImage,
    session: &Mutex<Session>,
    accumulator: &mut [f32],
    width: usize,
    height: usize,
    app_handle: &tauri::AppHandle,
    params: TileParams,
) -> Result<()> {
    let w = width as i32;
    let h = height as i32;
    let step = params.ucs.saturating_sub(params.overlap).max(1);
    let iperhl = (width.saturating_sub(params.ucs) as f64 / step as f64).ceil() as usize;
    let ipervl = (height.saturating_sub(params.ucs) as f64 / step as f64).ceil() as usize;
    let total = (iperhl + 1) * (ipervl + 1);

    for i in 0..total {
        let yi = i / (iperhl + 1);
        let xi = i % (iperhl + 1);
        let x0 =
            params.ucs as i32 * xi as i32 - params.overlap as i32 * xi as i32 - params.pad as i32;
        let y0 =
            params.ucs as i32 * yi as i32 - params.overlap as i32 * yi as i32 - params.pad as i32;

        if i % 10 == 0 {
            let pct = (i as f32 / total as f32) * 100.0;
            let _ = app_handle.emit("denoise-progress", format!("Denoising… {:.0}%", pct));
        }

        let crop = extract_tile_mirror(img, x0, y0, params.cs);
        let input_values = crop.as_standard_layout().to_owned();
        let t_input = Tensor::from_array(input_values)?;

        let out = {
            let mut sess = session.lock().unwrap();
            let outputs = sess.run(ort::inputs![t_input])?;
            let arr = outputs[0].try_extract_array::<f32>()?.to_owned();
            arr.into_dimensionality::<ndarray::Ix4>()
                .map_err(|e| anyhow::anyhow!("Unexpected output shape: {}", e))?
        };

        let x1pad = (0i32).max(x0 + params.cs as i32 - w) as usize;
        let y1pad = (0i32).max(y0 + params.cs as i32 - h) as usize;
        let ud0 = params.pad;
        let ud1 = params.pad;
        let ud2 = params.cs - params.pad.max(x1pad);
        let ud3 = params.cs - params.pad.max(y1pad);
        let absx0 = (x0 + params.pad as i32).max(0) as usize;
        let absy0 = (y0 + params.pad as i32).max(0) as usize;

        let mut tile = out;
        apply_seamless(
            &mut tile,
            &SeamlessBlend {
                ud0,
                ud1,
                ud2,
                ud3,
                absx0,
                absy0,
                fswidth: width,
                fsheight: height,
                overlap: params.overlap,
            },
        );

        for cy in 0..(ud3 - ud1) {
            for cx in 0..(ud2 - ud0) {
                let gx = absx0 + cx;
                let gy = absy0 + cy;
                if gx < width && gy < height {
                    let base = (gy * width + gx) * 3;
                    accumulator[base] += tile[[0, 0, ud1 + cy, ud0 + cx]].clamp(0.0, 1.0);
                    accumulator[base + 1] += tile[[0, 1, ud1 + cy, ud0 + cx]].clamp(0.0, 1.0);
                    accumulator[base + 2] += tile[[0, 2, ud1 + cy, ud0 + cx]].clamp(0.0, 1.0);
                }
            }
        }
    }
    Ok(())
}

fn accumulator_to_rgb32f(acc: &[f32], width: u32, height: u32) -> Rgb32FImage {
    let mut out = Rgb32FImage::new(width, height);
    for (i, p) in out.pixels_mut().enumerate() {
        let i3 = i * 3;
        *p = Rgb([
            acc[i3].clamp(0.0, 1.0),
            acc[i3 + 1].clamp(0.0, 1.0),
            acc[i3 + 2].clamp(0.0, 1.0),
        ]);
    }
    out
}

pub fn run_ai_denoise(
    rgb_img: &Rgb32FImage,
    intensity: f32,
    session: &Mutex<Session>,
    app_handle: &tauri::AppHandle,
) -> Result<DynamicImage> {
    let (width, height) = rgb_img.dimensions();
    let params = select_tile_params(intensity);

    let _ = app_handle.emit("denoise-progress", "Denoising (AI NIND)...");
    let mut accumulator = vec![0.0f32; width as usize * height as usize * 3];
    run_native_denoise(
        rgb_img,
        session,
        &mut accumulator,
        width as usize,
        height as usize,
        app_handle,
        params,
    )?;

    let out_img_buffer = accumulator_to_rgb32f(&accumulator, width, height);
    Ok(DynamicImage::ImageRgb32F(out_img_buffer))
}

pub fn run_lama_inpainting(
    image: &DynamicImage,
    mask: &GrayImage,
    lama_session: &Mutex<Session>,
) -> Result<RgbaImage> {
    let (w, h) = image.dimensions();

    let (mut min_x, mut min_y) = (w, h);
    let (mut max_x, mut max_y) = (0u32, 0u32);
    let mut has_mask = false;

    for (x, y, p) in mask.enumerate_pixels() {
        if p[0] > 0 {
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
            has_mask = true;
        }
    }

    if !has_mask {
        return Ok(image.to_rgba8());
    }

    let mask_w = max_x - min_x + 1;
    let mask_h = max_y - min_y + 1;

    let pad_x = 128.max((mask_w as f32 * 1.5) as u32);
    let pad_y = 128.max((mask_h as f32 * 1.5) as u32);

    let x0 = min_x.saturating_sub(pad_x);
    let y0 = min_y.saturating_sub(pad_y);
    let x1 = (max_x + pad_x).min(w.saturating_sub(1));
    let y1 = (max_y + pad_y).min(h.saturating_sub(1));

    let crop_w = x1 - x0 + 1;
    let crop_h = y1 - y0 + 1;

    let rgba = image.to_rgba8();

    let cropped_img = imageops::crop_imm(&rgba, x0, y0, crop_w, crop_h).to_image();
    let cropped_mask = imageops::crop_imm(mask, x0, y0, crop_w, crop_h).to_image();

    let max_dim_limit: u32 = 768;
    let needs_downscale = crop_w > max_dim_limit || crop_h > max_dim_limit;

    let (fw, fh, inf_img, inf_mask) = if needs_downscale {
        let scale = max_dim_limit as f32 / crop_w.max(crop_h) as f32;

        let scaled_w = (crop_w as f32 * scale).round().max(1.0) as u32;
        let scaled_h = (crop_h as f32 * scale).round().max(1.0) as u32;

        (
            scaled_w,
            scaled_h,
            imageops::resize(&cropped_img, scaled_w, scaled_h, FilterType::Lanczos3),
            imageops::resize(&cropped_mask, scaled_w, scaled_h, FilterType::Triangle),
        )
    } else {
        (crop_w, crop_h, cropped_img.clone(), cropped_mask.clone())
    };

    let align = 64u32;
    let mut tensor_dim = fw.max(fh);
    if tensor_dim % align != 0 {
        tensor_dim += align - (tensor_dim % align);
    }
    let tensor_dim = tensor_dim.max(align) as usize;

    let mut img_tensor = Array::<f32, _>::zeros((1, 3, tensor_dim, tensor_dim));
    let mut msk_tensor = Array::<f32, _>::zeros((1, 1, tensor_dim, tensor_dim));

    for y in 0..tensor_dim {
        for x in 0..tensor_dim {
            let sx = (x as u32).min(fw.saturating_sub(1));
            let sy = (y as u32).min(fh.saturating_sub(1));

            let p = inf_img.get_pixel(sx, sy);
            let m = inf_mask.get_pixel(sx, sy)[0];

            img_tensor[[0, 0, y, x]] = p[0] as f32 / 255.0;
            img_tensor[[0, 1, y, x]] = p[1] as f32 / 255.0;
            img_tensor[[0, 2, y, x]] = p[2] as f32 / 255.0;
            msk_tensor[[0, 0, y, x]] = if m > 0 { 1.0 } else { 0.0 };
        }
    }

    let t_img = Tensor::from_array(img_tensor.into_dyn().as_standard_layout().into_owned())?;
    let t_msk = Tensor::from_array(msk_tensor.into_dyn().as_standard_layout().into_owned())?;

    let output_tensor = {
        let mut session = lama_session.lock().unwrap();
        let outputs = session.run(ort::inputs!["image" => t_img, "mask" => t_msk])?;
        outputs[0].try_extract_array::<f32>()?.to_owned()
    };

    let mut result_inf = RgbaImage::new(fw, fh);
    for y in 0..fh {
        for x in 0..fw {
            let r = output_tensor[[0, 0, y as usize, x as usize]].clamp(0.0, 255.0) as u8;
            let g = output_tensor[[0, 1, y as usize, x as usize]].clamp(0.0, 255.0) as u8;
            let b = output_tensor[[0, 2, y as usize, x as usize]].clamp(0.0, 255.0) as u8;
            result_inf.put_pixel(x, y, Rgba([r, g, b, 255]));
        }
    }

    let result_crop = if needs_downscale {
        imageops::resize(&result_inf, crop_w, crop_h, FilterType::Lanczos3)
    } else {
        result_inf
    };

    let mut final_image = image.to_rgba8();

    for y in 0..crop_h {
        for x in 0..crop_w {
            let m = cropped_mask.get_pixel(x, y)[0];
            if m > 0 {
                let alpha = m as f32 / 255.0;
                let p = result_crop.get_pixel(x, y);
                let gx = x0 + x;
                let gy = y0 + y;
                let orig = final_image.get_pixel(gx, gy);

                let r = (p[0] as f32 * alpha + orig[0] as f32 * (1.0 - alpha)) as u8;
                let g = (p[1] as f32 * alpha + orig[1] as f32 * (1.0 - alpha)) as u8;
                let b = (p[2] as f32 * alpha + orig[2] as f32 * (1.0 - alpha)) as u8;

                final_image.put_pixel(gx, gy, Rgba([r, g, b, 255]));
            }
        }
    }

    Ok(final_image)
}

pub fn generate_image_embeddings(
    image: &DynamicImage,
    encoder: &Mutex<Session>,
) -> Result<ImageEmbeddings> {
    let (orig_width, orig_height) = image.dimensions();

    let long_side = orig_width.max(orig_height) as f32;
    let scale = SAM_INPUT_SIZE as f32 / long_side;
    let new_width = (orig_width as f32 * scale).round() as u32;
    let new_height = (orig_height as f32 * scale).round() as u32;

    let resized_image = image.resize(new_width, new_height, FilterType::Triangle);
    let rgb_image = resized_image.into_rgb8();
    let (actual_width, actual_height) = rgb_image.dimensions();
    let raw_pixels = rgb_image.as_raw();

    let mut input_tensor: Array<u8, _> =
        Array::zeros((1, 3, SAM_INPUT_SIZE as usize, SAM_INPUT_SIZE as usize));

    let w_usize = actual_width as usize;
    for y in 0..(actual_height as usize) {
        for x in 0..w_usize {
            let idx = (y * w_usize + x) * 3;
            input_tensor[[0, 0, y, x]] = raw_pixels[idx];
            input_tensor[[0, 1, y, x]] = raw_pixels[idx + 1];
            input_tensor[[0, 2, y, x]] = raw_pixels[idx + 2];
        }
    }

    let input_tensor_dyn = input_tensor.into_dyn();
    let input_values = input_tensor_dyn.as_standard_layout();
    let input_tensor_ort = Tensor::from_array(input_values.into_owned())?;
    let mut session = encoder.lock().unwrap();
    let outputs = session.run(ort::inputs![input_tensor_ort])?;

    let embeddings = outputs[0].try_extract_array::<f32>()?.to_owned();

    Ok(ImageEmbeddings {
        path_hash: "".to_string(),
        embeddings: embeddings.into_dyn(),
        original_size: (orig_width, orig_height),
    })
}

pub fn run_sam_decoder(
    decoder: &Mutex<Session>,
    embeddings: &ImageEmbeddings,
    start_point: (f64, f64),
    end_point: (f64, f64),
) -> Result<GrayImage> {
    let (orig_width, orig_height) = embeddings.original_size;
    let long_side = orig_width.max(orig_height) as f64;
    let scale = SAM_INPUT_SIZE as f64 / long_side;

    let iters = 2;

    let is_point =
        (start_point.0 - end_point.0).abs() < 1e-6 && (start_point.1 - end_point.1).abs() < 1e-6;
    let mut point_coords = Vec::new();
    let mut point_labels = Vec::new();

    if is_point {
        point_coords.push((
            (start_point.0 * scale) as f32,
            (start_point.1 * scale) as f32,
        ));
        point_labels.push(1.0f32);
    } else {
        let x1 = (start_point.0.min(end_point.0) * scale) as f32;
        let y1 = (start_point.1.min(end_point.1) * scale) as f32;
        let x2 = (start_point.0.max(end_point.0) * scale) as f32;
        let y2 = (start_point.1.max(end_point.1) * scale) as f32;
        point_coords.push((x1, y1));
        point_coords.push((x2, y2));
        point_labels.push(2.0f32);
        point_labels.push(3.0f32);
    }

    let mut mask_input = Array::zeros((1, 1, 256, 256)).into_dyn();
    let mut has_mask_input = 0.0f32;

    let orig_im_size =
        Array::from_shape_vec((2,), vec![orig_height as f32, orig_width as f32])?.into_dyn();

    let mut final_mask_data: Vec<u8> = Vec::new();
    let mut final_w = 0;
    let mut final_h = 0;

    for i in 0..iters {
        let pc_len = point_coords.len();
        let pl_len = point_labels.len();

        let coords_flat: Vec<f32> = point_coords.iter().flat_map(|&(x, y)| vec![x, y]).collect();
        let coords_array = Array::from_shape_vec((1, pc_len, 2), coords_flat)?.into_dyn();
        let labels_array = Array::from_shape_vec((1, pl_len), point_labels.clone())?.into_dyn();

        let t_embeddings = Tensor::from_array(
            embeddings
                .embeddings
                .clone()
                .as_standard_layout()
                .into_owned(),
        )?;
        let t_point_coords = Tensor::from_array(coords_array.as_standard_layout().into_owned())?;
        let t_point_labels = Tensor::from_array(labels_array.as_standard_layout().into_owned())?;
        let t_mask_input =
            Tensor::from_array(mask_input.clone().as_standard_layout().into_owned())?;
        let t_has_mask = Tensor::from_array(
            Array::from_elem((1,), has_mask_input)
                .into_dyn()
                .as_standard_layout()
                .into_owned(),
        )?;
        let t_orig_im_size =
            Tensor::from_array(orig_im_size.clone().as_standard_layout().into_owned())?;

        let mask_tensor = {
            let mut session = decoder.lock().unwrap();
            let outputs = session.run(ort::inputs![
                t_embeddings,
                t_point_coords,
                t_point_labels,
                t_mask_input,
                t_has_mask,
                t_orig_im_size
            ])?;
            outputs[0].try_extract_array::<f32>()?.to_owned()
        };

        let mask_dims = mask_tensor.shape();
        let h = mask_dims[2];
        let w = mask_dims[3];
        let area = h * w;

        let mask_slice = mask_tensor.as_slice().unwrap();
        let first_mask_slice = &mask_slice[0..area];

        if i == iters - 1 {
            final_mask_data = first_mask_slice
                .iter()
                .map(|&val| if val > 0.0 { 255 } else { 0 })
                .collect();
            final_w = w;
            final_h = h;
            break;
        }

        let mut binary_mask = vec![false; area];
        let mut mask_area = 0.0;
        let mut min_x = w;
        let mut min_y = h;
        let mut max_x = 0;
        let mut max_y = 0;

        for (idx, &val) in first_mask_slice.iter().enumerate() {
            if val > 0.0 {
                binary_mask[idx] = true;
                let x = idx % w;
                let y = idx / w;
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);
                mask_area += 1.0;
            }
        }

        if mask_area == 0.0 || min_x > max_x {
            final_mask_data = first_mask_slice
                .iter()
                .map(|&val| if val > 0.0 { 255 } else { 0 })
                .collect();
            final_w = w;
            final_h = h;
            break;
        }

        let dt_in = edt_2d(&binary_mask, w, h);
        let mut max_in = 0.0;
        let mut pos_idx = 0;
        for (idx, &v) in dt_in.iter().enumerate() {
            if v > max_in {
                max_in = v;
                pos_idx = idx;
            }
        }
        let pos_y = pos_idx / w;
        let pos_x = pos_idx % w;

        let mut rev_mask = vec![false; area];
        for (idx, is_true) in binary_mask.iter().enumerate() {
            rev_mask[idx] = !is_true;
        }
        let mut dt_out = edt_2d(&rev_mask, w, h);

        for y in 0..h {
            for x in 0..w {
                if x < min_x || x > max_x || y < min_y || y > max_y {
                    dt_out[y * w + x] = 0.0;
                }
            }
        }

        let mut max_out = 0.0;
        let mut neg_idx = 0;
        for (idx, &v) in dt_out.iter().enumerate() {
            if v > max_out {
                max_out = v;
                neg_idx = idx;
            }
        }
        let neg_y = neg_idx / w;
        let neg_x = neg_idx % w;

        point_coords.clear();
        point_labels.clear();

        point_coords.push(((pos_x as f64 * scale) as f32, (pos_y as f64 * scale) as f32));
        point_labels.push(1.0);
        point_coords.push(((neg_x as f64 * scale) as f32, (neg_y as f64 * scale) as f32));
        point_labels.push(0.0);
        point_coords.push(((min_x as f64 * scale) as f32, (min_y as f64 * scale) as f32));
        point_labels.push(2.0);
        point_coords.push(((max_x as f64 * scale) as f32, (max_y as f64 * scale) as f32));
        point_labels.push(3.0);

        let mut gaus_dt = vec![0.0f32; area];
        let variance = (mask_area / 4.0_f32).max(1.0_f32);
        for (idx, &is_true) in binary_mask.iter().enumerate() {
            if is_true {
                let diff = dt_in[idx] - max_in;
                gaus_dt[idx] = (-(diff * diff) / variance).exp();
            }
        }

        let mask_f32_vec: Vec<f32> = first_mask_slice
            .iter()
            .map(|&v| if v > 0.0 { 15.0 } else { -15.0 })
            .collect();

        let img_mask_f32 =
            ImageBuffer::<Luma<f32>, Vec<f32>>::from_raw(w as u32, h as u32, mask_f32_vec).unwrap();
        let img_gaus_f32 =
            ImageBuffer::<Luma<f32>, Vec<f32>>::from_raw(w as u32, h as u32, gaus_dt).unwrap();

        let resized_mask = imageops::resize(&img_mask_f32, 256, 256, FilterType::Triangle);
        let resized_gaus = imageops::resize(&img_gaus_f32, 256, 256, FilterType::Triangle);

        let rm_raw = resized_mask.as_raw();
        let rg_raw = resized_gaus.as_raw();
        let mut mask_input_flat = vec![0.0f32; 256 * 256];

        for i in 0..(256 * 256) {
            let m_val = rm_raw[i];
            let mut g_val = rg_raw[i];
            if g_val <= 0.0 {
                g_val = 1.0;
            }
            mask_input_flat[i] = m_val * g_val;
        }

        mask_input = Array::from_shape_vec((1, 1, 256, 256), mask_input_flat)
            .unwrap()
            .into_dyn();
        has_mask_input = 1.0;
    }

    let gray_mask = GrayImage::from_raw(final_w as u32, final_h as u32, final_mask_data)
        .ok_or_else(|| anyhow::anyhow!("Failed to create mask image from raw data"))?;

    let feathered_mask = image::imageops::blur(&gray_mask, 2.0);

    Ok(feathered_mask)
}

pub fn run_sky_seg_model(
    image: &DynamicImage,
    sky_seg_session: &Mutex<Session>,
) -> Result<GrayImage> {
    let (orig_width, orig_height) = image.dimensions();

    let resized_image = image.resize(SKYSEG_INPUT_SIZE, SKYSEG_INPUT_SIZE, FilterType::Triangle);
    let (resized_w, resized_h) = resized_image.dimensions();
    let resized_rgb = resized_image.into_rgb8();
    let raw_pixels = resized_rgb.as_raw();

    let paste_x = ((SKYSEG_INPUT_SIZE - resized_w) / 2) as usize;
    let paste_y = ((SKYSEG_INPUT_SIZE - resized_h) / 2) as usize;

    let mut input_tensor: Array<f32, _> =
        Array::zeros((1, 3, SKYSEG_INPUT_SIZE as usize, SKYSEG_INPUT_SIZE as usize));

    let mean = [0.485, 0.456, 0.406];
    let std = [0.229, 0.224, 0.225];

    let rw = resized_w as usize;
    let rh = resized_h as usize;

    for y in 0..rh {
        for x in 0..rw {
            let idx = (y * rw + x) * 3;
            let dest_y = y + paste_y;
            let dest_x = x + paste_x;

            input_tensor[[0, 0, dest_y, dest_x]] =
                (raw_pixels[idx] as f32 / 255.0 - mean[0]) / std[0];
            input_tensor[[0, 1, dest_y, dest_x]] =
                (raw_pixels[idx + 1] as f32 / 255.0 - mean[1]) / std[1];
            input_tensor[[0, 2, dest_y, dest_x]] =
                (raw_pixels[idx + 2] as f32 / 255.0 - mean[2]) / std[2];
        }
    }

    let input_tensor_dyn = input_tensor.into_dyn();
    let t_input = Tensor::from_array(input_tensor_dyn.as_standard_layout().into_owned())?;

    let mut session = sky_seg_session.lock().unwrap();
    let outputs = session.run(ort::inputs![t_input])?;
    let output_tensor = outputs[0].try_extract_array::<f32>()?.to_owned();
    let out_slice = output_tensor.as_slice().unwrap();

    let mut min_val = f32::MAX;
    let mut max_val = f32::MIN;
    for &v in out_slice {
        min_val = min_val.min(v);
        max_val = max_val.max(v);
    }

    let range = max_val - min_val;
    let scale = if range > 1e-6 { 255.0 / range } else { 0.0 };

    let usize_size = SKYSEG_INPUT_SIZE as usize;
    let mut cropped_mask_data = Vec::with_capacity(rw * rh);

    for y in 0..rh {
        let src_y = y + paste_y;
        for x in 0..rw {
            let src_x = x + paste_x;
            let val = out_slice[src_y * usize_size + src_x];
            let pixel = if range > 1e-6 {
                ((val - min_val) * scale) as u8
            } else {
                0
            };
            cropped_mask_data.push(pixel);
        }
    }

    let cropped_mask = GrayImage::from_raw(resized_w, resized_h, cropped_mask_data)
        .ok_or_else(|| anyhow::anyhow!("Failed to create mask from Sky Segmentation output"))?;

    let final_mask = imageops::resize(&cropped_mask, orig_width, orig_height, FilterType::Triangle);

    Ok(final_mask)
}

pub fn run_u2netp_model(
    image: &DynamicImage,
    u2netp_session: &Mutex<Session>,
) -> Result<GrayImage> {
    let (orig_width, orig_height) = image.dimensions();

    let resized_image = image.resize(U2NETP_INPUT_SIZE, U2NETP_INPUT_SIZE, FilterType::Triangle);
    let (resized_w, resized_h) = resized_image.dimensions();
    let resized_rgb = resized_image.into_rgb8();
    let raw_pixels = resized_rgb.as_raw();

    let paste_x = ((U2NETP_INPUT_SIZE - resized_w) / 2) as usize;
    let paste_y = ((U2NETP_INPUT_SIZE - resized_h) / 2) as usize;

    let mut input_tensor: Array<f32, _> =
        Array::zeros((1, 3, U2NETP_INPUT_SIZE as usize, U2NETP_INPUT_SIZE as usize));

    let mean = [0.485, 0.456, 0.406];
    let std = [0.229, 0.224, 0.225];

    let rw = resized_w as usize;
    let rh = resized_h as usize;

    for y in 0..rh {
        for x in 0..rw {
            let idx = (y * rw + x) * 3;
            let dest_y = y + paste_y;
            let dest_x = x + paste_x;

            input_tensor[[0, 0, dest_y, dest_x]] =
                (raw_pixels[idx] as f32 / 255.0 - mean[0]) / std[0];
            input_tensor[[0, 1, dest_y, dest_x]] =
                (raw_pixels[idx + 1] as f32 / 255.0 - mean[1]) / std[1];
            input_tensor[[0, 2, dest_y, dest_x]] =
                (raw_pixels[idx + 2] as f32 / 255.0 - mean[2]) / std[2];
        }
    }

    let input_tensor_dyn = input_tensor.into_dyn();
    let t_input = Tensor::from_array(input_tensor_dyn.as_standard_layout().into_owned())?;

    let mut session = u2netp_session.lock().unwrap();
    let outputs = session.run(ort::inputs![t_input])?;
    let output_tensor = outputs[0].try_extract_array::<f32>()?.to_owned();
    let out_slice = output_tensor.as_slice().unwrap();

    let mut min_val = f32::MAX;
    let mut max_val = f32::MIN;
    for &v in out_slice {
        min_val = min_val.min(v);
        max_val = max_val.max(v);
    }

    let range = max_val - min_val;
    let scale = if range > 1e-6 { 255.0 / range } else { 0.0 };

    let usize_size = U2NETP_INPUT_SIZE as usize;
    let mut cropped_mask_data = Vec::with_capacity(rw * rh);

    for y in 0..rh {
        let src_y = y + paste_y;
        for x in 0..rw {
            let src_x = x + paste_x;
            let val = out_slice[src_y * usize_size + src_x];
            let pixel = if range > 1e-6 {
                ((val - min_val) * scale) as u8
            } else {
                0
            };
            cropped_mask_data.push(pixel);
        }
    }

    let cropped_mask = GrayImage::from_raw(resized_w, resized_h, cropped_mask_data)
        .ok_or_else(|| anyhow::anyhow!("Failed to create mask from U-2-Netp output"))?;

    let final_mask = imageops::resize(&cropped_mask, orig_width, orig_height, FilterType::Triangle);

    Ok(final_mask)
}

pub fn run_depth_anything_model(
    image: &DynamicImage,
    depth_session: &Mutex<Session>,
) -> Result<GrayImage> {
    let resized_image = image.resize(DEPTH_INPUT_SIZE, DEPTH_INPUT_SIZE, FilterType::Triangle);
    let (resized_w, resized_h) = resized_image.dimensions();
    let resized_rgb = resized_image.into_rgb8();
    let raw_pixels = resized_rgb.as_raw();

    let paste_x = ((DEPTH_INPUT_SIZE - resized_w) / 2) as usize;
    let paste_y = ((DEPTH_INPUT_SIZE - resized_h) / 2) as usize;

    let mut input_tensor: Array<f32, _> =
        Array::zeros((1, 3, DEPTH_INPUT_SIZE as usize, DEPTH_INPUT_SIZE as usize));

    let mean = [0.485, 0.456, 0.406];
    let std = [0.229, 0.224, 0.225];

    let rw = resized_w as usize;
    let rh = resized_h as usize;

    for y in 0..rh {
        for x in 0..rw {
            let idx = (y * rw + x) * 3;
            let dest_y = y + paste_y;
            let dest_x = x + paste_x;

            input_tensor[[0, 0, dest_y, dest_x]] =
                (raw_pixels[idx] as f32 / 255.0 - mean[0]) / std[0];
            input_tensor[[0, 1, dest_y, dest_x]] =
                (raw_pixels[idx + 1] as f32 / 255.0 - mean[1]) / std[1];
            input_tensor[[0, 2, dest_y, dest_x]] =
                (raw_pixels[idx + 2] as f32 / 255.0 - mean[2]) / std[2];
        }
    }

    let input_tensor_dyn = input_tensor.into_dyn();
    let t_input = Tensor::from_array(input_tensor_dyn.as_standard_layout().into_owned())?;

    let mut session = depth_session.lock().unwrap();
    let outputs = session.run(ort::inputs![t_input])?;
    let output_tensor = outputs[0].try_extract_array::<f32>()?.to_owned();
    let out_slice = output_tensor.as_slice().unwrap();

    let usize_size = DEPTH_INPUT_SIZE as usize;

    let mut min_val = f32::MAX;
    let mut max_val = f32::MIN;
    for y in 0..rh {
        let src_y = y + paste_y;
        for x in 0..rw {
            let src_x = x + paste_x;
            let val = out_slice[src_y * usize_size + src_x];
            min_val = min_val.min(val);
            max_val = max_val.max(val);
        }
    }

    let range = max_val - min_val;
    let scale = if range > 1e-6 { 255.0 / range } else { 0.0 };

    let mut cropped_depth_data = Vec::with_capacity(rw * rh);

    for y in 0..rh {
        let src_y = y + paste_y;
        for x in 0..rw {
            let src_x = x + paste_x;
            let val = out_slice[src_y * usize_size + src_x];
            let pixel = if range > 1e-6 {
                ((val - min_val) * scale) as u8
            } else {
                0
            };
            cropped_depth_data.push(pixel);
        }
    }

    let depth_map = GrayImage::from_raw(resized_w, resized_h, cropped_depth_data)
        .ok_or_else(|| anyhow::anyhow!("Failed to create mask from Depth output"))?;

    Ok(depth_map)
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct AiSubjectMaskParameters {
    pub start_x: f64,
    pub start_y: f64,
    pub end_x: f64,
    pub end_y: f64,
    #[serde(default)]
    pub mask_data_base64: Option<String>,
    #[serde(default)]
    pub rotation: Option<f32>,
    #[serde(default)]
    pub flip_horizontal: Option<bool>,
    #[serde(default)]
    pub flip_vertical: Option<bool>,
    #[serde(default)]
    pub orientation_steps: Option<u8>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct AiSkyMaskParameters {
    #[serde(default)]
    pub mask_data_base64: Option<String>,
    #[serde(default)]
    pub rotation: Option<f32>,
    #[serde(default)]
    pub flip_horizontal: Option<bool>,
    #[serde(default)]
    pub flip_vertical: Option<bool>,
    #[serde(default)]
    pub orientation_steps: Option<u8>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct AiForegroundMaskParameters {
    #[serde(default)]
    pub mask_data_base64: Option<String>,
    #[serde(default)]
    pub rotation: Option<f32>,
    #[serde(default)]
    pub flip_horizontal: Option<bool>,
    #[serde(default)]
    pub flip_vertical: Option<bool>,
    #[serde(default)]
    pub orientation_steps: Option<u8>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct AiDepthMaskParameters {
    #[serde(default)]
    pub min_depth: f32,
    #[serde(default)]
    pub max_depth: f32,
    #[serde(default)]
    pub min_fade: f32,
    #[serde(default)]
    pub max_fade: f32,
    #[serde(default)]
    pub feather: f32,
    #[serde(default)]
    pub mask_data_base64: Option<String>,
    #[serde(default)]
    pub rotation: Option<f32>,
    #[serde(default)]
    pub flip_horizontal: Option<bool>,
    #[serde(default)]
    pub flip_vertical: Option<bool>,
    #[serde(default)]
    pub orientation_steps: Option<u8>,
}
