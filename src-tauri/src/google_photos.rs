use std::collections::HashMap;
use std::fs;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use chrono::Utc;
use once_cell::sync::Lazy;
use rand::RngExt;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tauri::{AppHandle, Manager};

use crate::app_settings::{AppSettings, load_settings, save_settings};
use crate::file_management::parse_virtual_path;

const AUTH_URL: &str = "https://accounts.google.com/o/oauth2/v2/auth";
const TOKEN_URL: &str = "https://oauth2.googleapis.com/token";
const PHOTOS_API: &str = "https://photoslibrary.googleapis.com/v1";
const UPLOAD_URL: &str = "https://photoslibrary.googleapis.com/v1/uploads";
const GOOGLE_PHOTOS_SCOPES: &[&str] = &[
    "https://www.googleapis.com/auth/photoslibrary.appendonly",
    "https://www.googleapis.com/auth/photoslibrary.readonly.appcreateddata",
    "https://www.googleapis.com/auth/photoslibrary.edit.appcreateddata",
];

static PENDING_OAUTH: Lazy<Mutex<HashMap<String, PendingOAuth>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

#[derive(Clone)]
struct PendingOAuth {
    code_verifier: String,
    redirect_uri: String,
    result: Arc<Mutex<Option<Result<String, String>>>>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GooglePhotosLoginStart {
    pub authorization_url: String,
    pub state: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GooglePhotosLoginPoll {
    pub complete: bool,
    pub authenticated: bool,
    pub message: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GooglePhotosStatus {
    pub authenticated: bool,
    pub album_id: Option<String>,
    pub album_title: String,
    pub synced_count: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct GooglePhotosSyncEntry {
    pub media_item_id: String,
    pub product_url: Option<String>,
    pub base_url: Option<String>,
    pub filename: String,
    pub synced_at: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GooglePhotosAlbum {
    pub id: String,
    pub title: String,
    pub product_url: Option<String>,
    pub media_items_count: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GooglePhotosMediaFile {
    pub path: String,
    pub rating: u8,
    pub tags: Option<Vec<String>>,
    pub exif: Option<HashMap<String, String>>,
    pub modified: i64,
    pub is_edited: bool,
    pub is_virtual_copy: bool,
    pub google_photos_media_id: String,
    pub google_photos_product_url: Option<String>,
    pub google_photos_base_url: Option<String>,
    pub filename: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GooglePhotosSyncResult {
    pub synced: Vec<String>,
    pub failed: Vec<GooglePhotosSyncFailure>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GooglePhotosSyncFailure {
    pub path: String,
    pub message: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct GooglePhotosToken {
    access_token: String,
    refresh_token: Option<String>,
    expires_at: i64,
    token_type: Option<String>,
    scope: Option<String>,
}

#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
    expires_in: Option<i64>,
    refresh_token: Option<String>,
    token_type: Option<String>,
    scope: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct AlbumResponse {
    id: String,
    title: Option<String>,
    product_url: Option<String>,
    media_items_count: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct MediaItemResponse {
    id: String,
    product_url: Option<String>,
    base_url: Option<String>,
    filename: Option<String>,
    media_metadata: Option<MediaMetadataResponse>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct MediaMetadataResponse {
    creation_time: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct SearchResponse {
    media_items: Option<Vec<MediaItemResponse>>,
    next_page_token: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct BatchCreateResponse {
    new_media_item_results: Vec<NewMediaItemResult>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct NewMediaItemResult {
    upload_token: Option<String>,
    status: Option<GoogleStatus>,
    media_item: Option<MediaItemResponse>,
}

#[derive(Deserialize)]
struct GoogleStatus {
    code: Option<i32>,
    message: Option<String>,
}

fn now_timestamp() -> i64 {
    Utc::now().timestamp()
}

fn app_data_file(app_handle: &AppHandle, filename: &str) -> Result<PathBuf, String> {
    let dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| e.to_string())?;
    if !dir.exists() {
        fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    }
    Ok(dir.join(filename))
}

fn token_path(app_handle: &AppHandle) -> Result<PathBuf, String> {
    app_data_file(app_handle, "google_photos_tokens.json")
}

fn sync_index_path(app_handle: &AppHandle) -> Result<PathBuf, String> {
    app_data_file(app_handle, "google_photos_sync_index.json")
}

fn read_token(app_handle: &AppHandle) -> Result<Option<GooglePhotosToken>, String> {
    let path = token_path(app_handle)?;
    if !path.exists() {
        return Ok(None);
    }
    let content = fs::read_to_string(path).map_err(|e| e.to_string())?;
    serde_json::from_str(&content)
        .map(Some)
        .map_err(|e| format!("Failed to read Google Photos token: {}", e))
}

fn write_token(app_handle: &AppHandle, token: &GooglePhotosToken) -> Result<(), String> {
    let path = token_path(app_handle)?;
    let content = serde_json::to_string_pretty(token).map_err(|e| e.to_string())?;
    fs::write(path, content).map_err(|e| e.to_string())
}

fn read_sync_index(
    app_handle: &AppHandle,
) -> Result<HashMap<String, GooglePhotosSyncEntry>, String> {
    let path = sync_index_path(app_handle)?;
    if !path.exists() {
        return Ok(HashMap::new());
    }
    let content = fs::read_to_string(path).map_err(|e| e.to_string())?;
    serde_json::from_str(&content)
        .map_err(|e| format!("Failed to read Google Photos sync index: {}", e))
}

fn write_sync_index(
    app_handle: &AppHandle,
    index: &HashMap<String, GooglePhotosSyncEntry>,
) -> Result<(), String> {
    let path = sync_index_path(app_handle)?;
    let content = serde_json::to_string_pretty(index).map_err(|e| e.to_string())?;
    fs::write(path, content).map_err(|e| e.to_string())
}

fn random_urlsafe_string(byte_count: usize) -> String {
    let mut rng = rand::rng();
    let bytes: Vec<u8> = (0..byte_count).map(|_| rng.random::<u8>()).collect();
    URL_SAFE_NO_PAD.encode(bytes)
}

fn pkce_challenge(verifier: &str) -> String {
    let digest = Sha256::digest(verifier.as_bytes());
    URL_SAFE_NO_PAD.encode(digest)
}

fn percent_encode(input: &str) -> String {
    let mut encoded = String::new();
    for byte in input.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                encoded.push(byte as char)
            }
            b' ' => encoded.push_str("%20"),
            _ => encoded.push_str(&format!("%{:02X}", byte)),
        }
    }
    encoded
}

fn percent_decode(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut output = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%'
            && i + 2 < bytes.len()
            && let Ok(hex) = std::str::from_utf8(&bytes[i + 1..i + 3])
            && let Ok(value) = u8::from_str_radix(hex, 16)
        {
            output.push(value);
            i += 3;
            continue;
        }
        output.push(if bytes[i] == b'+' { b' ' } else { bytes[i] });
        i += 1;
    }
    String::from_utf8_lossy(&output).to_string()
}

fn form_body(fields: &[(&str, String)]) -> String {
    fields
        .iter()
        .map(|(key, value)| format!("{}={}", percent_encode(key), percent_encode(value)))
        .collect::<Vec<_>>()
        .join("&")
}

fn parse_query(path: &str) -> HashMap<String, String> {
    let query = path.split_once('?').map(|(_, q)| q).unwrap_or("");
    query
        .split('&')
        .filter_map(|pair| {
            let (key, value) = pair.split_once('=')?;
            Some((percent_decode(key), percent_decode(value)))
        })
        .collect()
}

fn send_oauth_browser_response(stream: &mut std::net::TcpStream, title: &str, body: &str) {
    let html = format!(
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>{}</title></head><body style=\"font-family:system-ui,sans-serif;margin:2rem;\"><h1>{}</h1><p>{}</p></body></html>",
        title, title, body
    );
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        html.len(),
        html
    );
    let _ = stream.write_all(response.as_bytes());
}

fn format_oauth_error(error: &str, description: Option<&str>) -> String {
    let trimmed_description = description.map(str::trim).filter(|value| !value.is_empty());
    match (error, trimmed_description) {
        ("access_denied", Some(description)) => {
            format!("Google sign-in was denied: {}", description)
        }
        ("access_denied", None) => {
            "Google sign-in was denied or blocked. If Google shows \"Access blocked\", add this account as an OAuth test user or complete OAuth verification in Google Cloud, then try again.".to_string()
        }
        (_, Some(description)) => format!("Google sign-in failed: {} ({})", description, error),
        (_, None) => format!("Google sign-in failed: {}", error),
    }
}

fn start_loopback_listener(
    expected_state: String,
    result: Arc<Mutex<Option<Result<String, String>>>>,
) -> Result<String, String> {
    let listener = TcpListener::bind("127.0.0.1:0")
        .map_err(|e| format!("Failed to start OAuth listener: {}", e))?;
    let port = listener.local_addr().map_err(|e| e.to_string())?.port();
    listener
        .set_nonblocking(false)
        .map_err(|e| format!("Failed to configure OAuth listener: {}", e))?;

    std::thread::spawn(move || {
        let _ = listener.set_ttl(1);
        if let Ok((mut stream, _)) = listener.accept() {
            let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
            let mut buffer = [0_u8; 4096];
            let read = stream.read(&mut buffer).unwrap_or(0);
            let request = String::from_utf8_lossy(&buffer[..read]);
            let first_line = request.lines().next().unwrap_or("");
            let path = first_line.split_whitespace().nth(1).unwrap_or("");
            let params = parse_query(path);

            let response_result =
                match (params.get("state"), params.get("code"), params.get("error")) {
                    (Some(state), Some(code), _) if state == &expected_state => {
                        send_oauth_browser_response(
                            &mut stream,
                            "RapidRAW Google Photos login complete",
                            "You can close this browser tab and return to RapidRAW.",
                        );
                        Ok(code.to_string())
                    }
                    (Some(state), _, Some(error)) if state == &expected_state => {
                        let message = format_oauth_error(
                            error,
                            params.get("error_description").map(String::as_str),
                        );
                        send_oauth_browser_response(
                            &mut stream,
                            "RapidRAW Google Photos login was not completed",
                            &message,
                        );
                        Err(message)
                    }
                    _ => {
                        send_oauth_browser_response(
                            &mut stream,
                            "RapidRAW Google Photos login failed",
                            "The OAuth response did not match the request that RapidRAW started.",
                        );
                        Err("OAuth state mismatch".to_string())
                    }
                };

            if let Ok(mut lock) = result.lock() {
                *lock = Some(response_result);
            }
        }
    });

    Ok(format!("http://127.0.0.1:{}/oauth2redirect", port))
}

fn auth_config(settings: &AppSettings) -> Result<(String, Option<String>), String> {
    let client_id = settings
        .google_photos_client_id
        .as_deref()
        .unwrap_or("")
        .trim()
        .to_string();
    if client_id.is_empty() {
        return Err("Enter a Google OAuth desktop client ID first.".to_string());
    }
    let client_secret = settings
        .google_photos_client_secret
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToString::to_string);
    Ok((client_id, client_secret))
}

async fn exchange_code_for_token(
    app_handle: &AppHandle,
    settings: &AppSettings,
    code: &str,
    code_verifier: &str,
    redirect_uri: &str,
) -> Result<(), String> {
    let (client_id, client_secret) = auth_config(settings)?;
    let client = reqwest::Client::new();
    let mut form = vec![
        ("client_id", client_id),
        ("code", code.to_string()),
        ("code_verifier", code_verifier.to_string()),
        ("grant_type", "authorization_code".to_string()),
        ("redirect_uri", redirect_uri.to_string()),
    ];
    if let Some(secret) = client_secret {
        form.push(("client_secret", secret));
    }

    let response = client
        .post(TOKEN_URL)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .body(form_body(&form))
        .send()
        .await
        .map_err(|e| format!("Token exchange failed: {}", e))?;
    if !response.status().is_success() {
        return Err(format!(
            "Token exchange failed: {}",
            response.text().await.unwrap_or_default()
        ));
    }
    let token_response: TokenResponse = response.json().await.map_err(|e| e.to_string())?;
    let token = GooglePhotosToken {
        access_token: token_response.access_token,
        refresh_token: token_response.refresh_token,
        expires_at: now_timestamp() + token_response.expires_in.unwrap_or(3600),
        token_type: token_response.token_type,
        scope: token_response.scope,
    };
    write_token(app_handle, &token)
}

async fn get_access_token(
    app_handle: &AppHandle,
    settings: &AppSettings,
) -> Result<String, String> {
    let mut token = read_token(app_handle)?.ok_or("Connect to Google Photos first.".to_string())?;
    if token.expires_at > now_timestamp() + 60 {
        return Ok(token.access_token);
    }

    let refresh_token = token
        .refresh_token
        .clone()
        .ok_or("Google Photos login expired. Disconnect and sign in again.".to_string())?;
    let (client_id, client_secret) = auth_config(settings)?;
    let client = reqwest::Client::new();
    let mut form = vec![
        ("client_id", client_id),
        ("grant_type", "refresh_token".to_string()),
        ("refresh_token", refresh_token),
    ];
    if let Some(secret) = client_secret {
        form.push(("client_secret", secret));
    }

    let response = client
        .post(TOKEN_URL)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .body(form_body(&form))
        .send()
        .await
        .map_err(|e| format!("Token refresh failed: {}", e))?;
    if !response.status().is_success() {
        return Err(format!(
            "Token refresh failed: {}",
            response.text().await.unwrap_or_default()
        ));
    }
    let refreshed: TokenResponse = response.json().await.map_err(|e| e.to_string())?;
    token.access_token = refreshed.access_token;
    token.expires_at = now_timestamp() + refreshed.expires_in.unwrap_or(3600);
    token.token_type = refreshed.token_type.or(token.token_type);
    token.scope = refreshed.scope.or(token.scope);
    write_token(app_handle, &token)?;
    Ok(token.access_token)
}

async fn google_json_error(response: reqwest::Response) -> String {
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    if body.is_empty() {
        status.to_string()
    } else {
        format!("{}: {}", status, body)
    }
}

fn album_title(settings: &AppSettings) -> String {
    settings
        .google_photos_album_title
        .as_deref()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or("RapidRaw")
        .trim()
        .to_string()
}

async fn ensure_album_id(settings: &AppSettings) -> Result<String, String> {
    settings
        .google_photos_album_id
        .clone()
        .filter(|id| !id.trim().is_empty())
        .ok_or("Create or select a RapidRaw Google Photos album first.".to_string())
}

fn filename_for_path(path: &str) -> String {
    path.split("?vc=")
        .next()
        .unwrap_or(path)
        .split(['/', '\\'])
        .next_back()
        .unwrap_or("RapidRaw photo")
        .to_string()
}

fn mime_for_path(path: &str) -> String {
    match path
        .split("?vc=")
        .next()
        .unwrap_or(path)
        .split('.')
        .next_back()
        .unwrap_or("")
        .to_ascii_lowercase()
        .as_str()
    {
        "jpg" | "jpeg" => "image/jpeg",
        "png" => "image/png",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "heic" | "heif" => "image/heic",
        "tif" | "tiff" => "image/tiff",
        "mp4" => "video/mp4",
        "mov" => "video/quicktime",
        _ => "application/octet-stream",
    }
    .to_string()
}

#[tauri::command]
pub fn google_photos_start_login(app_handle: AppHandle) -> Result<GooglePhotosLoginStart, String> {
    let settings = load_settings(app_handle.clone())?;
    let (client_id, _) = auth_config(&settings)?;
    let state = random_urlsafe_string(32);
    let verifier = random_urlsafe_string(64);
    let challenge = pkce_challenge(&verifier);
    let result = Arc::new(Mutex::new(None));
    let redirect_uri = start_loopback_listener(state.clone(), result.clone())?;
    let scope = GOOGLE_PHOTOS_SCOPES.join(" ");
    let authorization_url = format!(
        "{}?response_type=code&client_id={}&redirect_uri={}&scope={}&state={}&code_challenge={}&code_challenge_method=S256&access_type=offline&prompt=consent",
        AUTH_URL,
        percent_encode(&client_id),
        percent_encode(&redirect_uri),
        percent_encode(&scope),
        percent_encode(&state),
        percent_encode(&challenge)
    );

    PENDING_OAUTH.lock().unwrap().insert(
        state.clone(),
        PendingOAuth {
            code_verifier: verifier,
            redirect_uri,
            result,
        },
    );

    Ok(GooglePhotosLoginStart {
        authorization_url,
        state,
    })
}

#[tauri::command]
pub async fn google_photos_poll_login(
    app_handle: AppHandle,
    state: String,
) -> Result<GooglePhotosLoginPoll, String> {
    let pending = {
        let pending_map = PENDING_OAUTH.lock().unwrap();
        pending_map.get(&state).cloned()
    }
    .ok_or("No active Google Photos login.".to_string())?;

    let received = pending.result.lock().unwrap().clone();
    match received {
        None => Ok(GooglePhotosLoginPoll {
            complete: false,
            authenticated: false,
            message: "Waiting for Google sign-in.".to_string(),
        }),
        Some(Err(error)) => {
            PENDING_OAUTH.lock().unwrap().remove(&state);
            Ok(GooglePhotosLoginPoll {
                complete: true,
                authenticated: false,
                message: error,
            })
        }
        Some(Ok(code)) => {
            let settings = load_settings(app_handle.clone())?;
            exchange_code_for_token(
                &app_handle,
                &settings,
                &code,
                &pending.code_verifier,
                &pending.redirect_uri,
            )
            .await?;
            PENDING_OAUTH.lock().unwrap().remove(&state);
            Ok(GooglePhotosLoginPoll {
                complete: true,
                authenticated: true,
                message: "Connected to Google Photos.".to_string(),
            })
        }
    }
}

#[tauri::command]
pub fn google_photos_cancel_login(state: String) -> Result<(), String> {
    PENDING_OAUTH.lock().unwrap().remove(&state);
    Ok(())
}

#[tauri::command]
pub fn google_photos_get_status(app_handle: AppHandle) -> Result<GooglePhotosStatus, String> {
    let settings = load_settings(app_handle.clone())?;
    let authenticated = read_token(&app_handle)?.is_some();
    let synced_count = read_sync_index(&app_handle)?.len();
    Ok(GooglePhotosStatus {
        authenticated,
        album_id: settings.google_photos_album_id.clone(),
        album_title: album_title(&settings),
        synced_count,
    })
}

#[tauri::command]
pub fn google_photos_get_sync_index(
    app_handle: AppHandle,
) -> Result<HashMap<String, GooglePhotosSyncEntry>, String> {
    read_sync_index(&app_handle)
}

#[tauri::command]
pub fn google_photos_disconnect(app_handle: AppHandle) -> Result<(), String> {
    let token_path = token_path(&app_handle)?;
    if token_path.exists() {
        fs::remove_file(token_path).map_err(|e| e.to_string())?;
    }
    let mut settings = load_settings(app_handle.clone())?;
    settings.google_photos_integration_enabled = Some(false);
    save_settings(settings, app_handle)?;
    Ok(())
}

#[tauri::command]
pub async fn google_photos_create_album(
    app_handle: AppHandle,
    title: Option<String>,
) -> Result<GooglePhotosAlbum, String> {
    let mut settings = load_settings(app_handle.clone())?;
    let access_token = get_access_token(&app_handle, &settings).await?;
    let requested_title = title
        .as_deref()
        .filter(|s| !s.trim().is_empty())
        .map(str::trim)
        .unwrap_or("RapidRaw")
        .to_string();

    let client = reqwest::Client::new();
    let response = client
        .post(format!("{}/albums", PHOTOS_API))
        .bearer_auth(access_token)
        .json(&json!({ "album": { "title": requested_title } }))
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !response.status().is_success() {
        return Err(format!(
            "Failed to create Google Photos album: {}",
            google_json_error(response).await
        ));
    }
    let album: AlbumResponse = response.json().await.map_err(|e| e.to_string())?;
    settings.google_photos_album_id = Some(album.id.clone());
    settings.google_photos_album_title = Some(
        album
            .title
            .clone()
            .unwrap_or_else(|| requested_title.clone()),
    );
    settings.google_photos_integration_enabled = Some(true);
    save_settings(settings, app_handle)?;
    Ok(GooglePhotosAlbum {
        id: album.id,
        title: album.title.unwrap_or(requested_title),
        product_url: album.product_url,
        media_items_count: album.media_items_count,
    })
}

#[tauri::command]
pub async fn google_photos_rename_album(
    app_handle: AppHandle,
    title: String,
) -> Result<GooglePhotosAlbum, String> {
    let mut settings = load_settings(app_handle.clone())?;
    let album_id = ensure_album_id(&settings).await?;
    let access_token = get_access_token(&app_handle, &settings).await?;
    let next_title = title.trim();
    if next_title.is_empty() {
        return Err("Album title cannot be empty.".to_string());
    }

    let client = reqwest::Client::new();
    let response = client
        .patch(format!(
            "{}/albums/{}?updateMask=title",
            PHOTOS_API,
            percent_encode(&album_id)
        ))
        .bearer_auth(access_token)
        .json(&json!({ "id": album_id, "title": next_title }))
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !response.status().is_success() {
        return Err(format!(
            "Failed to rename Google Photos album: {}",
            google_json_error(response).await
        ));
    }
    let album: AlbumResponse = response.json().await.map_err(|e| e.to_string())?;
    settings.google_photos_album_title = Some(
        album
            .title
            .clone()
            .unwrap_or_else(|| next_title.to_string()),
    );
    save_settings(settings, app_handle)?;
    Ok(GooglePhotosAlbum {
        id: album.id,
        title: album.title.unwrap_or_else(|| next_title.to_string()),
        product_url: album.product_url,
        media_items_count: album.media_items_count,
    })
}

#[tauri::command]
pub async fn google_photos_list_album_media(
    app_handle: AppHandle,
) -> Result<Vec<GooglePhotosMediaFile>, String> {
    let settings = load_settings(app_handle.clone())?;
    let album_id = ensure_album_id(&settings).await?;
    let access_token = get_access_token(&app_handle, &settings).await?;
    let client = reqwest::Client::new();
    let mut page_token: Option<String> = None;
    let mut items = Vec::new();

    loop {
        let mut body = json!({ "albumId": album_id, "pageSize": 100 });
        if let Some(token) = page_token.as_deref() {
            body["pageToken"] = json!(token);
        }
        let response = client
            .post(format!("{}/mediaItems:search", PHOTOS_API))
            .bearer_auth(&access_token)
            .json(&body)
            .send()
            .await
            .map_err(|e| e.to_string())?;
        if !response.status().is_success() {
            return Err(format!(
                "Failed to list Google Photos album: {}",
                google_json_error(response).await
            ));
        }
        let page: SearchResponse = response.json().await.map_err(|e| e.to_string())?;
        if let Some(media_items) = page.media_items {
            for item in media_items {
                let filename = item.filename.unwrap_or_else(|| item.id.clone());
                let modified = item
                    .media_metadata
                    .as_ref()
                    .and_then(|metadata| metadata.creation_time.as_deref())
                    .and_then(|time| chrono::DateTime::parse_from_rfc3339(time).ok())
                    .map(|time| time.timestamp())
                    .unwrap_or_else(now_timestamp);
                items.push(GooglePhotosMediaFile {
                    path: format!("googlephotos://{}/{}", item.id, percent_encode(&filename)),
                    rating: 0,
                    tags: Some(vec!["googlephotos:synced".to_string()]),
                    exif: None,
                    modified,
                    is_edited: false,
                    is_virtual_copy: false,
                    google_photos_media_id: item.id,
                    google_photos_product_url: item.product_url,
                    google_photos_base_url: item.base_url.map(|url| format!("{}=w600-h600", url)),
                    filename,
                });
            }
        }
        page_token = page.next_page_token;
        if page_token.is_none() {
            break;
        }
    }

    Ok(items)
}

#[tauri::command]
pub async fn google_photos_sync_files(
    app_handle: AppHandle,
    paths: Vec<String>,
) -> Result<GooglePhotosSyncResult, String> {
    let settings = load_settings(app_handle.clone())?;
    let album_id = ensure_album_id(&settings).await?;
    let access_token = get_access_token(&app_handle, &settings).await?;
    let client = reqwest::Client::new();
    let mut upload_tokens: Vec<(String, String)> = Vec::new();
    let mut failed = Vec::new();

    for path in paths {
        let (physical_path, _) = parse_virtual_path(&path);
        match fs::read(&physical_path) {
            Ok(bytes) => {
                let mime = mime_for_path(&path);
                let response = client
                    .post(UPLOAD_URL)
                    .bearer_auth(&access_token)
                    .header("Content-Type", "application/octet-stream")
                    .header("X-Goog-Upload-Content-Type", mime)
                    .header("X-Goog-Upload-Protocol", "raw")
                    .body(bytes)
                    .send()
                    .await;

                match response {
                    Ok(resp) if resp.status().is_success() => match resp.text().await {
                        Ok(token) if !token.trim().is_empty() => upload_tokens.push((path, token)),
                        Ok(_) => failed.push(GooglePhotosSyncFailure {
                            path,
                            message: "Google Photos returned an empty upload token.".to_string(),
                        }),
                        Err(err) => failed.push(GooglePhotosSyncFailure {
                            path,
                            message: err.to_string(),
                        }),
                    },
                    Ok(resp) => failed.push(GooglePhotosSyncFailure {
                        path,
                        message: google_json_error(resp).await,
                    }),
                    Err(err) => failed.push(GooglePhotosSyncFailure {
                        path,
                        message: err.to_string(),
                    }),
                }
            }
            Err(err) => failed.push(GooglePhotosSyncFailure {
                path,
                message: format!("Failed to read local file: {}", err),
            }),
        }
    }

    let mut synced = Vec::new();
    let mut sync_index = read_sync_index(&app_handle)?;
    for chunk in upload_tokens.chunks(50) {
        let new_media_items: Vec<Value> = chunk
            .iter()
            .map(|(path, upload_token)| {
                json!({
                    "simpleMediaItem": {
                        "fileName": filename_for_path(path),
                        "uploadToken": upload_token,
                    }
                })
            })
            .collect();
        let response = client
            .post(format!("{}/mediaItems:batchCreate", PHOTOS_API))
            .bearer_auth(&access_token)
            .json(&json!({ "albumId": album_id, "newMediaItems": new_media_items }))
            .send()
            .await
            .map_err(|e| e.to_string())?;
        if !response.status().is_success() {
            let message = google_json_error(response).await;
            for (path, _) in chunk {
                failed.push(GooglePhotosSyncFailure {
                    path: path.clone(),
                    message: message.clone(),
                });
            }
            continue;
        }

        let batch: BatchCreateResponse = response.json().await.map_err(|e| e.to_string())?;
        for result in batch.new_media_item_results {
            let upload_token = result.upload_token.unwrap_or_default();
            let path = chunk
                .iter()
                .find(|(_, token)| token == &upload_token)
                .map(|(path, _)| path.clone())
                .unwrap_or_default();
            if path.is_empty() {
                continue;
            }
            if let Some(media_item) = result.media_item {
                sync_index.insert(
                    path.clone(),
                    GooglePhotosSyncEntry {
                        media_item_id: media_item.id,
                        product_url: media_item.product_url,
                        base_url: media_item.base_url.map(|url| format!("{}=w600-h600", url)),
                        filename: media_item
                            .filename
                            .unwrap_or_else(|| filename_for_path(&path)),
                        synced_at: Utc::now().to_rfc3339(),
                    },
                );
                synced.push(path);
            } else {
                let message = result
                    .status
                    .and_then(|status| {
                        status
                            .message
                            .or_else(|| status.code.map(|code| code.to_string()))
                    })
                    .unwrap_or_else(|| "Google Photos did not create this media item.".to_string());
                failed.push(GooglePhotosSyncFailure { path, message });
            }
        }
    }

    write_sync_index(&app_handle, &sync_index)?;
    Ok(GooglePhotosSyncResult { synced, failed })
}

#[tauri::command]
pub async fn google_photos_unsync_files(
    app_handle: AppHandle,
    paths: Vec<String>,
) -> Result<GooglePhotosSyncResult, String> {
    let settings = load_settings(app_handle.clone())?;
    let album_id = ensure_album_id(&settings).await?;
    let access_token = get_access_token(&app_handle, &settings).await?;
    let client = reqwest::Client::new();
    let mut sync_index = read_sync_index(&app_handle)?;
    let mut media_ids = Vec::new();
    let mut path_by_media_id = HashMap::new();
    let mut failed = Vec::new();

    for path in paths {
        let media_id = if let Some(id) = path.strip_prefix("googlephotos://") {
            id.split('/').next().map(ToString::to_string)
        } else {
            sync_index
                .get(&path)
                .map(|entry| entry.media_item_id.clone())
        };

        if let Some(media_id) = media_id {
            path_by_media_id.insert(media_id.clone(), path);
            media_ids.push(media_id);
        } else {
            failed.push(GooglePhotosSyncFailure {
                path,
                message: "This photo is not tracked as synced to Google Photos.".to_string(),
            });
        }
    }

    let mut synced = Vec::new();
    for chunk in media_ids.chunks(50) {
        let response = client
            .post(format!(
                "{}/albums/{}:batchRemoveMediaItems",
                PHOTOS_API,
                percent_encode(&album_id)
            ))
            .bearer_auth(&access_token)
            .json(&json!({ "mediaItemIds": chunk }))
            .send()
            .await
            .map_err(|e| e.to_string())?;
        if !response.status().is_success() {
            let message = google_json_error(response).await;
            for media_id in chunk {
                failed.push(GooglePhotosSyncFailure {
                    path: path_by_media_id
                        .get(media_id)
                        .cloned()
                        .unwrap_or_else(|| media_id.clone()),
                    message: message.clone(),
                });
            }
            continue;
        }

        for media_id in chunk {
            if let Some(path) = path_by_media_id.get(media_id) {
                sync_index.retain(|entry_path, entry| {
                    entry_path != path && entry.media_item_id != *media_id
                });
                synced.push(path.clone());
            }
        }
    }

    write_sync_index(&app_handle, &sync_index)?;
    Ok(GooglePhotosSyncResult { synced, failed })
}
