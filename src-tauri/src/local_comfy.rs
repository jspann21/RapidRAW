use crate::app_settings::LocalAiGenerationSettings;
use std::fs;
use std::io::{Cursor, Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow};
use image::{DynamicImage, GrayImage, ImageFormat, Rgba, RgbaImage};
use reqwest::{Client, multipart};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tauri::Emitter;
use uuid::Uuid;

#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x08000000;

const RUNTIME_URL: &str = "https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_nvidia.7z";
const CROP_AND_STITCH_URL: &str =
    "https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch/archive/refs/heads/main.zip";
const CONNECT_TIMEOUT: Duration = Duration::from_secs(120);
const GENERATION_TIMEOUT: Duration = Duration::from_secs(300);

const WORKFLOW_JSON: &str = r#"
{
  "1": { "inputs": { "ckpt_name": "XL_RealVisXL_V5.0_Lightning.safetensors" }, "class_type": "CheckpointLoaderSimple" },
  "7": { "inputs": { "text": "", "clip": [ "1", 1 ] }, "class_type": "CLIPTextEncode" },
  "8": { "inputs": { "text": "", "clip": [ "1", 1 ] }, "class_type": "CLIPTextEncode" },
  "9": { "inputs": { "width": [ "37", 0 ], "height": [ "37", 0 ], "batch_size": 1, "color": 0 }, "class_type": "EmptyImage" },
  "10": { "inputs": { "value": 0.5, "mask": [ "36", 2 ] }, "class_type": "ThresholdMask" },
  "11": { "inputs": { "x": 0, "y": 0, "resize_source": false, "destination": [ "36", 1 ], "source": [ "9", 0 ], "mask": [ "10", 0 ] }, "class_type": "ImageCompositeMasked" },
  "12": { "inputs": { "control_net_name": "diffusion_pytorch_model_promax.safetensors" }, "class_type": "ControlNetLoader" },
  "13": { "inputs": { "type": "repaint", "control_net": [ "12", 0 ] }, "class_type": "SetUnionControlNetType" },
  "14": { "inputs": { "strength": 1, "start_percent": 0, "end_percent": 1, "positive": [ "7", 0 ], "negative": [ "8", 0 ], "control_net": [ "13", 0 ], "image": [ "11", 0 ] }, "class_type": "ControlNetApplyAdvanced" },
  "15": { "inputs": { "pixels": [ "36", 1 ], "vae": [ "49", 0 ] }, "class_type": "VAEEncode" },
  "16": { "inputs": { "samples": [ "15", 0 ], "mask": [ "36", 2 ] }, "class_type": "SetLatentNoiseMask" },
  "23": { "inputs": { "samples": [ "28", 0 ], "vae": [ "49", 0 ] }, "class_type": "VAEDecode" },
  "28": { "inputs": { "seed": 0, "steps": 8, "cfg": 1, "sampler_name": "euler", "scheduler": "ddim_uniform", "denoise": 1, "model": [ "1", 0 ], "positive": [ "14", 0 ], "negative": [ "14", 1 ], "latent_image": [ "16", 0 ] }, "class_type": "KSampler" },
  "30": { "inputs": { "image": "" }, "class_type": "LoadImage" },
  "35": { "inputs": { "stitcher": [ "36", 0 ], "inpainted_image": [ "23", 0 ] }, "class_type": "InpaintStitchImproved" },
  "36": { "inputs": { "device_mode": "gpu (much faster)", "downscale_algorithm": "bilinear", "upscale_algorithm": "bicubic", "preresize": false, "preresize_mode": "ensure maximum resolution", "preresize_min_width": 1024, "preresize_min_height": 1024, "preresize_max_width": 16384, "preresize_max_height": 16384, "mask_fill_holes": false, "mask_expand_pixels": 0, "mask_invert": false, "mask_blend_pixels": 32, "mask_hipass_filter": 0.1, "extend_for_outpainting": false, "extend_up_factor": 1, "extend_down_factor": 1, "extend_left_factor": 1, "extend_right_factor": 1, "context_from_mask_extend_factor": 1.5, "output_resize_to_target_size": true, "output_target_width": [ "37", 0 ], "output_target_height": [ "37", 0 ], "output_padding": "32", "image": [ "30", 0 ], "mask": [ "48", 0 ] }, "class_type": "InpaintCropImproved" },
  "37": { "inputs": { "value": 1280 }, "class_type": "PrimitiveInt" },
  "41": { "inputs": { "images": [ "35", 0 ] }, "class_type": "PreviewImage" },
  "47": { "inputs": { "image": "" }, "class_type": "LoadImage" },
  "48": { "inputs": { "mask": [ "47", 1 ] }, "class_type": "InvertMask" },
  "49": { "inputs": { "vae_name": "sdxl_vae.safetensors" }, "class_type": "VAELoader" }
}
"#;

#[derive(Serialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct LocalComfyStatus {
    pub runtime_dir: String,
    pub runtime_installed: bool,
    pub custom_nodes_installed: bool,
    pub running: bool,
    pub port: Option<u16>,
    pub generative_ready: bool,
    pub last_error: Option<String>,
}

pub struct LocalComfyProcess {
    pub child: Child,
    pub port: u16,
}

impl Drop for LocalComfyProcess {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[derive(Clone)]
pub struct GenerativeAssetSpec {
    pub id: &'static str,
    pub name: &'static str,
    pub relative_path: &'static str,
    pub url: &'static str,
    pub sha256: &'static str,
    pub required: bool,
}

pub const GENERATIVE_ASSETS: &[GenerativeAssetSpec] = &[
    GenerativeAssetSpec {
        id: "comfy-sdxl-checkpoint",
        name: "RealVisXL Lightning SDXL",
        relative_path: "comfy/checkpoints/XL_RealVisXL_V5.0_Lightning.safetensors",
        url: "https://huggingface.co/ModelsLab/RealVisXL_V5.0_Lightning/resolve/main/RealVisXL_V5.0_Lightning_fp16.safetensors?download=true",
        sha256: "fabcadd9330dcc4f9702063428d40b9d4d07168d8acefc819b8d1d9db466b3ec",
        required: true,
    },
    GenerativeAssetSpec {
        id: "comfy-controlnet-union-promax",
        name: "ControlNet Union ProMax SDXL",
        relative_path: "comfy/controlnet/diffusion_pytorch_model_promax.safetensors",
        url: "https://huggingface.co/xinsir/controlnet-union-sdxl-1.0/resolve/main/diffusion_pytorch_model_promax.safetensors?download=true",
        sha256: "9fae2e50cb431bfcbe05822b59ec2228df545ef27f711dea8949e9f4ed9f7cdc",
        required: true,
    },
    GenerativeAssetSpec {
        id: "comfy-sdxl-vae",
        name: "SDXL VAE",
        relative_path: "comfy/vae/sdxl_vae.safetensors",
        url: "https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors?download=true",
        sha256: "63aeecb90ff7bc1c115395962d3e803571385b61938377bc7089b36e81e92e2e",
        required: true,
    },
];

fn url_component(value: &str) -> String {
    let mut encoded = String::with_capacity(value.len());
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                encoded.push(byte as char);
            }
            _ => encoded.push_str(&format!("%{byte:02X}")),
        }
    }
    encoded
}

#[derive(Deserialize)]
struct PromptResponse {
    prompt_id: String,
}

#[derive(Deserialize)]
struct UploadResponse {
    name: String,
    subfolder: Option<String>,
    #[serde(rename = "type")]
    image_type: Option<String>,
}

pub fn runtime_base_dir() -> Result<PathBuf> {
    let exe = std::env::current_exe()?;
    let install_dir = exe
        .parent()
        .ok_or_else(|| anyhow!("Could not resolve RapidRAW executable directory"))?;
    let dir = install_dir.join("local-ai-runtime");
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

fn comfy_portable_dir() -> Result<PathBuf> {
    Ok(runtime_base_dir()?
        .join("comfyui")
        .join("ComfyUI_windows_portable"))
}

fn comfy_root_dir() -> Result<PathBuf> {
    Ok(comfy_portable_dir()?.join("ComfyUI"))
}

fn python_exe() -> Result<PathBuf> {
    Ok(comfy_portable_dir()?
        .join("python_embeded")
        .join("python.exe"))
}

fn work_dir() -> Result<PathBuf> {
    let dir = runtime_base_dir()?.join("work");
    fs::create_dir_all(dir.join("input"))?;
    fs::create_dir_all(dir.join("output"))?;
    fs::create_dir_all(dir.join("temp"))?;
    Ok(dir)
}

pub fn status(models_dir: &Path, process: &Mutex<Option<LocalComfyProcess>>) -> LocalComfyStatus {
    let runtime_dir =
        comfy_portable_dir().unwrap_or_else(|_| PathBuf::from("local-ai-runtime/comfyui"));
    let runtime_installed = python_exe().is_ok_and(|path| path.exists())
        && comfy_root_dir().is_ok_and(|path| path.join("main.py").exists());
    let custom_nodes_installed = comfy_root_dir().is_ok_and(|path| {
        path.join("custom_nodes")
            .join("ComfyUI-Inpaint-CropAndStitch")
            .exists()
    });
    let running = process
        .lock()
        .ok()
        .and_then(|guard| guard.as_ref().map(|proc| proc.port))
        .is_some();
    let port = process
        .lock()
        .ok()
        .and_then(|guard| guard.as_ref().map(|proc| proc.port));
    let generative_models_ready = GENERATIVE_ASSETS
        .iter()
        .filter(|asset| asset.required)
        .all(|asset| models_dir.join(asset.relative_path).exists());

    LocalComfyStatus {
        runtime_dir: runtime_dir.to_string_lossy().to_string(),
        runtime_installed,
        custom_nodes_installed,
        running,
        port,
        generative_ready: runtime_installed && custom_nodes_installed && generative_models_ready,
        last_error: None,
    }
}

pub async fn download_runtime(app_handle: &tauri::AppHandle) -> Result<LocalComfyStatus> {
    if !cfg!(all(target_os = "windows", target_arch = "x86_64")) {
        return Err(anyhow!(
            "Local SDXL runtime is currently supported on Windows x64 only."
        ));
    }

    let base = runtime_base_dir()?;
    let archive_path = base.join("ComfyUI_windows_portable_nvidia.7z");
    let extract_root = base.join("comfyui");
    if python_exe().is_ok_and(|path| path.exists()) {
        install_custom_nodes(app_handle).await?;
        return Ok(status(
            &crate::ai_processing::get_models_dir(app_handle)?,
            &Mutex::new(None),
        ));
    }

    download_file_with_progress(
        app_handle,
        "ComfyUI Portable Runtime",
        RUNTIME_URL,
        &archive_path,
    )
    .await?;
    let _ = app_handle.emit(
        "local-ai-task-progress",
        json!({"task": "runtime", "message": "Extracting ComfyUI runtime...", "indeterminate": true}),
    );
    fs::create_dir_all(&extract_root)?;
    sevenz_rust2::decompress_file(&archive_path, &extract_root)
        .map_err(|e| anyhow!("Failed to extract ComfyUI runtime: {}", e))?;
    install_custom_nodes(app_handle).await?;
    let _ = app_handle.emit(
        "local-ai-task-progress",
        json!({"task": "runtime", "message": "ComfyUI runtime installed.", "done": true}),
    );
    Ok(status(
        &crate::ai_processing::get_models_dir(app_handle)?,
        &Mutex::new(None),
    ))
}

pub fn delete_runtime(
    app_handle: &tauri::AppHandle,
    process: &Mutex<Option<LocalComfyProcess>>,
) -> Result<LocalComfyStatus> {
    stop_runtime(process)?;
    let base = runtime_base_dir()?;
    let target = base.join("comfyui");
    if target.exists() {
        fs::remove_dir_all(&target)?;
    }
    let archive = base.join("ComfyUI_windows_portable_nvidia.7z");
    if archive.exists() {
        fs::remove_file(archive)?;
    }
    Ok(status(
        &crate::ai_processing::get_models_dir(app_handle)?,
        process,
    ))
}

async fn install_custom_nodes(app_handle: &tauri::AppHandle) -> Result<()> {
    let comfy_root = comfy_root_dir()?;
    let custom_nodes = comfy_root.join("custom_nodes");
    let target = custom_nodes.join("ComfyUI-Inpaint-CropAndStitch");
    if target.exists() {
        return Ok(());
    }
    fs::create_dir_all(&custom_nodes)?;
    let zip_path = runtime_base_dir()?.join("ComfyUI-Inpaint-CropAndStitch.zip");
    download_file_with_progress(
        app_handle,
        "Inpaint Crop & Stitch Nodes",
        CROP_AND_STITCH_URL,
        &zip_path,
    )
    .await?;
    let temp = runtime_base_dir()?.join("crop-and-stitch-extract");
    if temp.exists() {
        fs::remove_dir_all(&temp)?;
    }
    fs::create_dir_all(&temp)?;
    extract_zip(&zip_path, &temp)?;
    let extracted = fs::read_dir(&temp)?
        .flatten()
        .find(|entry| entry.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .map(|entry| entry.path())
        .ok_or_else(|| anyhow!("Custom node archive did not contain a folder"))?;
    fs::rename(extracted, &target)?;
    let _ = fs::remove_dir_all(&temp);
    Ok(())
}

fn extract_zip(zip_path: &Path, dest: &Path) -> Result<()> {
    let file = fs::File::open(zip_path)?;
    let mut archive = zip::ZipArchive::new(file)?;
    for index in 0..archive.len() {
        let mut entry = archive.by_index(index)?;
        let Some(enclosed) = entry.enclosed_name().map(|path| dest.join(path)) else {
            continue;
        };
        if entry.is_dir() {
            fs::create_dir_all(enclosed)?;
        } else {
            if let Some(parent) = enclosed.parent() {
                fs::create_dir_all(parent)?;
            }
            let mut out = fs::File::create(enclosed)?;
            std::io::copy(&mut entry, &mut out)?;
        }
    }
    Ok(())
}

pub async fn download_generative_assets(
    app_handle: &tauri::AppHandle,
    models_dir: &Path,
) -> Result<()> {
    for asset in GENERATIVE_ASSETS {
        download_generative_asset(app_handle, models_dir, asset.id).await?;
    }
    Ok(())
}

pub async fn download_generative_asset(
    app_handle: &tauri::AppHandle,
    models_dir: &Path,
    asset_id: &str,
) -> Result<()> {
    let asset = GENERATIVE_ASSETS
        .iter()
        .find(|asset| asset.id == asset_id)
        .ok_or_else(|| anyhow!("Unknown generative asset: {}", asset_id))?;
    let dest = models_dir.join(asset.relative_path);
    if verify_sha256(&dest, asset.sha256)? {
        return Ok(());
    }
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }
    download_file_with_progress(app_handle, asset.name, asset.url, &dest).await?;
    if !verify_sha256(&dest, asset.sha256)? {
        let _ = fs::remove_file(&dest);
        return Err(anyhow!(
            "Downloaded {} but SHA-256 verification failed.",
            asset.name
        ));
    }
    Ok(())
}

pub fn delete_generative_asset(models_dir: &Path, asset_id: &str) -> Result<()> {
    let asset = GENERATIVE_ASSETS
        .iter()
        .find(|asset| asset.id == asset_id)
        .ok_or_else(|| anyhow!("Unknown generative asset: {}", asset_id))?;
    let path = models_dir.join(asset.relative_path);
    let canonical_dir = models_dir.canonicalize()?;
    if path.exists() {
        let canonical_asset = path.canonicalize()?;
        if !canonical_asset.starts_with(canonical_dir) {
            return Err(anyhow!(
                "Refusing to delete model outside RapidRAW model directory"
            ));
        }
        fs::remove_file(canonical_asset)?;
    }
    Ok(())
}

async fn download_file_with_progress(
    app_handle: &tauri::AppHandle,
    name: &str,
    url: &str,
    dest: &Path,
) -> Result<()> {
    let mut response = reqwest::get(url).await?.error_for_status()?;
    let total_bytes = response.content_length();
    let temp = dest.with_extension("download");
    let mut file = fs::File::create(&temp)?;
    let mut downloaded_bytes = 0_u64;

    let _ = app_handle.emit(
        "ai-model-download-progress",
        json!({"modelName": name, "downloadedBytes": 0_u64, "totalBytes": total_bytes}),
    );
    while let Some(chunk) = response.chunk().await? {
        file.write_all(&chunk)?;
        downloaded_bytes += chunk.len() as u64;
        let _ = app_handle.emit(
            "ai-model-download-progress",
            json!({"modelName": name, "downloadedBytes": downloaded_bytes, "totalBytes": total_bytes}),
        );
    }
    file.flush()?;
    if dest.exists() {
        fs::remove_file(dest)?;
    }
    fs::rename(temp, dest)?;
    Ok(())
}

fn verify_sha256(path: &Path, expected: &str) -> Result<bool> {
    if expected.is_empty() || !path.exists() {
        return Ok(false);
    }
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 8192];
    loop {
        let bytes_read = file.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }
    Ok(hex::encode(hasher.finalize()) == expected)
}

pub async fn start_runtime(
    app_handle: &tauri::AppHandle,
    models_dir: &Path,
    process: &Mutex<Option<LocalComfyProcess>>,
) -> Result<u16> {
    if let Some(port) = process.lock().unwrap().as_ref().map(|proc| proc.port) {
        return Ok(port);
    }
    let runtime = status(models_dir, process);
    if !runtime.runtime_installed {
        return Err(anyhow!("ComfyUI runtime is not installed."));
    }
    if !runtime.custom_nodes_installed {
        return Err(anyhow!(
            "ComfyUI inpaint crop/stitch nodes are not installed."
        ));
    }
    write_extra_model_paths(models_dir)?;
    let work = work_dir()?;
    let port = free_port()?;
    let mut cmd = Command::new(python_exe()?);
    cmd.current_dir(comfy_portable_dir()?);
    cmd.arg("-s")
        .arg(comfy_root_dir()?.join("main.py"))
        .arg("--listen")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(port.to_string())
        .arg("--disable-auto-launch")
        .arg("--windows-standalone-build")
        .arg("--input-directory")
        .arg(work.join("input"))
        .arg("--output-directory")
        .arg(work.join("output"))
        .arg("--temp-directory")
        .arg(work.join("temp"))
        .arg("--extra-model-paths-config")
        .arg(comfy_root_dir()?.join("extra_model_paths.yaml"))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    #[cfg(target_os = "windows")]
    cmd.creation_flags(CREATE_NO_WINDOW);

    let child = cmd
        .spawn()
        .map_err(|e| anyhow!("Failed to start ComfyUI runtime: {}", e))?;
    *process.lock().unwrap() = Some(LocalComfyProcess { child, port });
    wait_for_runtime(port).await.map_err(|e| {
        let _ = stop_runtime(process);
        e
    })?;
    let _ = app_handle.emit(
        "local-ai-task-progress",
        json!({"task": "runtime", "message": "Local SDXL runtime is running.", "done": true}),
    );
    Ok(port)
}

pub fn stop_runtime(process: &Mutex<Option<LocalComfyProcess>>) -> Result<()> {
    if let Some(mut proc) = process.lock().unwrap().take() {
        let _ = proc.child.kill();
        let _ = proc.child.wait();
    }
    Ok(())
}

fn write_extra_model_paths(models_dir: &Path) -> Result<()> {
    let comfy_root = comfy_root_dir()?;
    fs::create_dir_all(&comfy_root)?;
    let base = models_dir
        .join("comfy")
        .to_string_lossy()
        .replace('\\', "/");
    let config = format!(
        "rapidraw:\n  base_path: {}\n  checkpoints: checkpoints\n  controlnet: controlnet\n  vae: vae\n",
        base
    );
    fs::write(comfy_root.join("extra_model_paths.yaml"), config)?;
    Ok(())
}

fn free_port() -> Result<u16> {
    let listener = TcpListener::bind(("127.0.0.1", 0))?;
    Ok(listener.local_addr()?.port())
}

async fn wait_for_runtime(port: u16) -> Result<()> {
    let client = Client::new();
    let start = Instant::now();
    while start.elapsed() < CONNECT_TIMEOUT {
        if client
            .get(format!("http://127.0.0.1:{}/system_stats", port))
            .send()
            .await
            .is_ok_and(|res| res.status().is_success())
        {
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(750)).await;
    }
    Err(anyhow!(
        "Timed out waiting for local SDXL runtime to start."
    ))
}

async fn upload_image(
    client: &Client,
    port: u16,
    filename: &str,
    image: &DynamicImage,
) -> Result<String> {
    let mut buf = Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(image.to_rgba8()).write_to(&mut buf, ImageFormat::Png)?;
    let part = multipart::Part::bytes(buf.into_inner())
        .file_name(filename.to_string())
        .mime_str("image/png")?;
    let form = multipart::Form::new()
        .text("type", "input")
        .text("overwrite", "true")
        .part("image", part);
    let res = client
        .post(format!("http://127.0.0.1:{}/upload/image", port))
        .multipart(form)
        .send()
        .await?
        .error_for_status()?
        .json::<UploadResponse>()
        .await?;
    Ok(match (res.subfolder, res.image_type) {
        (Some(subfolder), Some(image_type)) if !subfolder.is_empty() => {
            format!("{}/{} [{}]", subfolder, res.name, image_type)
        }
        (_, Some(image_type)) => format!("{} [{}]", res.name, image_type),
        _ => res.name,
    })
}

fn mask_to_alpha_image(mask: &GrayImage) -> DynamicImage {
    let mut image = RgbaImage::new(mask.width(), mask.height());
    for (x, y, pixel) in mask.enumerate_pixels() {
        image.put_pixel(x, y, Rgba([255, 255, 255, pixel[0]]));
    }
    DynamicImage::ImageRgba8(image)
}

fn build_workflow(
    source_name: &str,
    mask_name: &str,
    prompt: &str,
    settings: &LocalAiGenerationSettings,
) -> Result<Value> {
    let mut workflow: Value = serde_json::from_str(WORKFLOW_JSON)?;
    let seed = settings
        .seed
        .unwrap_or_else(|| rand::random::<u32>() as i64);
    workflow["7"]["inputs"]["text"] = json!(prompt);
    workflow["8"]["inputs"]["text"] = json!(settings.negative_prompt);
    workflow["14"]["inputs"]["strength"] = json!(settings.controlnet_strength.clamp(0.0, 2.0));
    workflow["28"]["inputs"]["seed"] = json!(seed);
    workflow["28"]["inputs"]["steps"] = json!(settings.steps.clamp(1, 60));
    workflow["28"]["inputs"]["cfg"] = json!(settings.cfg.clamp(0.0, 20.0));
    workflow["28"]["inputs"]["sampler_name"] = json!(settings.sampler_name);
    workflow["28"]["inputs"]["scheduler"] = json!(settings.scheduler);
    workflow["28"]["inputs"]["denoise"] = json!(settings.denoise.clamp(0.0, 1.0));
    workflow["30"]["inputs"]["image"] = json!(source_name);
    workflow["36"]["inputs"]["mask_blend_pixels"] = json!(settings.mask_blend_pixels.clamp(0, 128));
    workflow["47"]["inputs"]["image"] = json!(mask_name);
    workflow["37"]["inputs"]["value"] = json!(settings.crop_target.clamp(512, 2048));
    Ok(workflow)
}

pub async fn process_inpainting(
    app_handle: &tauri::AppHandle,
    models_dir: &Path,
    process: &Mutex<Option<LocalComfyProcess>>,
    source_image: &DynamicImage,
    mask: &GrayImage,
    prompt: String,
    generation_settings: &LocalAiGenerationSettings,
) -> Result<RgbaImage> {
    let runtime = status(models_dir, process);
    if !runtime.generative_ready {
        return Err(anyhow!(
            "Local GPU generative setup is incomplete. Install the SDXL runtime and required models in Settings."
        ));
    }
    let was_running = process.lock().unwrap().is_some();
    let port = start_runtime(app_handle, models_dir, process).await?;
    let client = Client::new();
    let result = async {
        let client_id = Uuid::new_v4().to_string();
        let source_name = format!("rapidraw-source-{}.png", client_id);
        let mask_name = format!("rapidraw-mask-{}.png", client_id);
        let source_ref = upload_image(&client, port, &source_name, source_image).await?;
        let mask_ref = upload_image(&client, port, &mask_name, &mask_to_alpha_image(mask)).await?;
        let workflow = build_workflow(&source_ref, &mask_ref, &prompt, generation_settings)?;
        let prompt_response = client
            .post(format!("http://127.0.0.1:{}/prompt", port))
            .json(&json!({"prompt": workflow, "client_id": client_id}))
            .send()
            .await?
            .error_for_status()?
            .json::<PromptResponse>()
            .await?;
        fetch_prompt_result(&client, port, &prompt_response.prompt_id).await
    }
    .await;
    if !was_running {
        let _ = stop_runtime(process);
    }
    result
}

async fn fetch_prompt_result(client: &Client, port: u16, prompt_id: &str) -> Result<RgbaImage> {
    let started = Instant::now();
    while started.elapsed() < GENERATION_TIMEOUT {
        let history: Value = client
            .get(format!("http://127.0.0.1:{}/history/{}", port, prompt_id))
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        if let Some(images) = history
            .get(prompt_id)
            .and_then(|item| item.get("outputs"))
            .and_then(|outputs| outputs.get("41"))
            .and_then(|node| node.get("images"))
            .and_then(|images| images.as_array())
            && let Some(image) = images.first()
        {
            let filename = image
                .get("filename")
                .and_then(|v| v.as_str())
                .unwrap_or_default();
            let subfolder = image
                .get("subfolder")
                .and_then(|v| v.as_str())
                .unwrap_or_default();
            let image_type = image.get("type").and_then(|v| v.as_str()).unwrap_or("temp");
            let view_url = format!(
                "http://127.0.0.1:{}/view?filename={}&subfolder={}&type={}",
                port,
                url_component(filename),
                url_component(subfolder),
                url_component(image_type)
            );
            let bytes = client
                .get(view_url)
                .send()
                .await?
                .error_for_status()?
                .bytes()
                .await?;
            return Ok(image::load_from_memory(&bytes)?.to_rgba8());
        }
        tokio::time::sleep(Duration::from_millis(750)).await;
    }
    Err(anyhow!("Timed out waiting for Local GPU SDXL generation."))
}

pub async fn run_self_test(
    app_handle: &tauri::AppHandle,
    models_dir: &Path,
    process: &Mutex<Option<LocalComfyProcess>>,
    generation_settings: &LocalAiGenerationSettings,
) -> Result<String> {
    let mut source = RgbaImage::new(256, 256);
    for (x, y, pixel) in source.enumerate_pixels_mut() {
        let shade = if (x / 32 + y / 32) % 2 == 0 { 180 } else { 220 };
        *pixel = Rgba([shade, shade, shade, 255]);
    }
    let mut mask = GrayImage::new(256, 256);
    for y in 96..160 {
        for x in 96..160 {
            mask.put_pixel(x, y, image::Luma([255]));
        }
    }
    let output = process_inpainting(
        app_handle,
        models_dir,
        process,
        &DynamicImage::ImageRgba8(source),
        &mask,
        "neutral studio background".to_string(),
        generation_settings,
    )
    .await?;
    if output.dimensions() != (256, 256) {
        return Err(anyhow!(
            "Local GPU SDXL self-test returned an unexpected image size."
        ));
    }
    Ok("Local GPU SDXL generative self-test completed.".to_string())
}
