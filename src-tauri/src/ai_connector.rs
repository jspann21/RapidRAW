use anyhow::{Result, anyhow};
use base64::{Engine as _, engine::general_purpose};
use image::{
    DynamicImage, GenericImageView, ImageFormat, RgbaImage, codecs::jpeg::JpegEncoder, imageops,
};
use reqwest::{Client, multipart};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Cursor;
use std::net::{IpAddr, SocketAddr, ToSocketAddrs};
use std::path::Path;
use std::time::{Duration, SystemTime};

#[derive(Serialize)]
struct InpaintRequest {
    source_id: String,
    prompt: String,
    negative_prompt: String,
    mask_image_base64: String,
    seed: i64,
}

#[derive(Deserialize)]
struct MiddlewareResponse {
    x: u32,
    y: u32,
    color: String,
}

fn is_forbidden_connector_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => {
            ip.is_unspecified()
                || ip.is_broadcast()
                || ip.is_documentation()
                || ip.is_link_local()
                || ip.is_multicast()
                || ip.octets() == [169, 254, 169, 254]
        }
        IpAddr::V6(ip) => {
            ip.is_unspecified() || ip.is_multicast() || ip.segments()[0] & 0xffc0 == 0xfe80
        }
    }
}

fn parse_connector_socket(address: &str) -> Result<(String, u16)> {
    let trimmed = address.trim().trim_end_matches('/');
    if trimmed.is_empty() {
        return Err(anyhow!("AI Connector address cannot be empty"));
    }
    if trimmed.contains("://") {
        return Err(anyhow!(
            "AI Connector address must be a host and port, for example 127.0.0.1:8188"
        ));
    }
    if trimmed
        .chars()
        .any(|ch| ch.is_whitespace() || matches!(ch, '/' | '\\' | '?' | '#' | '@'))
    {
        return Err(anyhow!("AI Connector address contains invalid characters"));
    }

    if trimmed.starts_with('[') {
        let socket: SocketAddr = trimmed
            .parse()
            .map_err(|_| anyhow!("AI Connector IPv6 address must look like [::1]:8188"))?;
        return Ok((socket.ip().to_string(), socket.port()));
    }

    let (host, port_str) = trimmed
        .rsplit_once(':')
        .ok_or_else(|| anyhow!("AI Connector address must include a port"))?;
    if host.is_empty() {
        return Err(anyhow!("AI Connector host cannot be empty"));
    }
    if !host
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '.'))
    {
        return Err(anyhow!("AI Connector host contains invalid characters"));
    }
    let port = port_str
        .parse::<u16>()
        .map_err(|_| anyhow!("AI Connector port must be between 1 and 65535"))?;
    if port == 0 {
        return Err(anyhow!("AI Connector port must be between 1 and 65535"));
    }

    Ok((host.to_string(), port))
}

pub fn validate_address_format(address: &str) -> Result<String> {
    let (host, port) = parse_connector_socket(address)?;
    if let Ok(ip) = host.parse::<IpAddr>()
        && is_forbidden_connector_ip(ip)
    {
        return Err(anyhow!(
            "AI Connector address points to a blocked network range"
        ));
    }

    if host.eq_ignore_ascii_case("metadata.google.internal") {
        return Err(anyhow!(
            "AI Connector address points to a blocked metadata host"
        ));
    }

    Ok(if host.contains(':') {
        format!("[{}]:{}", host, port)
    } else {
        format!("{}:{}", host, port)
    })
}

fn validate_address_for_request(address: &str) -> Result<String> {
    let normalized = validate_address_format(address)?;
    let mut resolved = normalized
        .to_socket_addrs()
        .map_err(|e| anyhow!("Failed to resolve AI Connector address: {}", e))?
        .peekable();

    if resolved.peek().is_none() {
        return Err(anyhow!("AI Connector address did not resolve"));
    }

    for socket in resolved {
        if is_forbidden_connector_ip(socket.ip()) {
            return Err(anyhow!(
                "AI Connector address resolves to a blocked network range"
            ));
        }
    }

    Ok(normalized)
}

pub fn base_url_for_address(address: &str) -> Result<String> {
    let normalized = validate_address_for_request(address)?;
    Ok(format!("http://{}", normalized))
}

fn http_client() -> Result<Client> {
    Client::builder()
        .timeout(Duration::from_secs(30))
        .connect_timeout(Duration::from_secs(5))
        .build()
        .map_err(Into::into)
}

pub fn generate_source_id(path_str: &str) -> Result<String> {
    let path = Path::new(path_str);
    let metadata = fs::metadata(path)?;
    let mod_time = metadata
        .modified()
        .unwrap_or(SystemTime::UNIX_EPOCH)
        .duration_since(SystemTime::UNIX_EPOCH)?
        .as_secs();

    let mut hasher = blake3::Hasher::new();
    hasher.update(path_str.as_bytes());
    hasher.update(&mod_time.to_le_bytes());
    Ok(hasher.finalize().to_hex().to_string())
}

fn image_to_base64(img: &DynamicImage) -> Result<String> {
    let mut buf = Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(img.to_rgba8()).write_to(&mut buf, ImageFormat::Png)?;
    Ok(general_purpose::STANDARD.encode(buf.get_ref()))
}

fn image_to_jpeg_bytes(img: &DynamicImage, quality: u8) -> Result<Vec<u8>> {
    let mut buf = Cursor::new(Vec::new());
    let mut encoder = JpegEncoder::new_with_quality(&mut buf, quality);
    encoder.encode_image(&img.to_rgb8())?;
    Ok(buf.into_inner())
}

async fn upload_source_image(
    client: &Client,
    base_url: &str,
    source_id: &str,
    image: &DynamicImage,
    token: Option<&str>,
) -> Result<()> {
    let jpeg_bytes = image_to_jpeg_bytes(image, 95)?;

    let part = multipart::Part::bytes(jpeg_bytes)
        .file_name("source.jpg")
        .mime_str("image/jpeg")?;

    let form = multipart::Form::new()
        .text("source_id", source_id.to_string())
        .part("file", part);

    let mut req = client
        .post(format!("{}/upload_source", base_url))
        .multipart(form);

    if let Some(auth_token) = token {
        req = req.bearer_auth(auth_token);
    }

    let res = req.send().await?;

    if !res.status().is_success() {
        return Err(anyhow!("Upload failed: {}", res.text().await?));
    }
    Ok(())
}

fn composite_full_res(
    response: MiddlewareResponse,
    full_width: u32,
    full_height: u32,
) -> Result<RgbaImage> {
    let crop_color_bytes = general_purpose::STANDARD.decode(&response.color)?;
    let crop_color = image::load_from_memory(&crop_color_bytes)?;

    let mut full_color = RgbaImage::new(full_width, full_height);
    imageops::overlay(
        &mut full_color,
        &crop_color,
        response.x.into(),
        response.y.into(),
    );

    Ok(full_color)
}

pub async fn check_status(address: &str) -> Result<bool> {
    let base_url = base_url_for_address(address)?;
    let client = http_client()?;
    let res = client.get(format!("{}/health", base_url)).send().await;
    Ok(res.is_ok())
}

pub async fn process_inpainting(
    base_url: &str,
    source_path: &str,
    full_source_image: &DynamicImage,
    mask_image: &DynamicImage,
    prompt: String,
    token: Option<&str>,
) -> Result<RgbaImage> {
    let client = http_client()?;
    let source_id = generate_source_id(source_path)?;
    let mask_b64 = image_to_base64(mask_image)?;
    let (w, h) = full_source_image.dimensions();

    let payload = InpaintRequest {
        source_id: source_id.clone(),
        prompt,
        negative_prompt: "blur, low quality, distortion, watermark".to_string(),
        mask_image_base64: mask_b64,
        seed: 0,
    };

    let url = format!("{}/inpaint", base_url);

    let mut req = client.post(&url).json(&payload);
    if let Some(auth_token) = token {
        req = req.bearer_auth(auth_token);
    }

    let response = req.send().await?;

    let middleware_data: MiddlewareResponse = if response.status() == 404 {
        upload_source_image(&client, base_url, &source_id, full_source_image, token).await?;

        let mut retry_req = client.post(&url).json(&payload);
        if let Some(auth_token) = token {
            retry_req = retry_req.bearer_auth(auth_token);
        }

        let retry_res = retry_req.send().await?;
        if !retry_res.status().is_success() {
            return Err(anyhow!(
                "AI generation failed after upload: {}",
                retry_res.text().await?
            ));
        }
        retry_res.json().await?
    } else if !response.status().is_success() {
        return Err(anyhow!("AI generation failed: {}", response.text().await?));
    } else {
        response.json().await?
    };

    composite_full_res(middleware_data, w, h)
}

#[cfg(test)]
mod tests {
    use super::validate_address_format;

    #[test]
    fn accepts_host_port_addresses() {
        assert_eq!(
            validate_address_format("127.0.0.1:8188").unwrap(),
            "127.0.0.1:8188"
        );
        assert_eq!(
            validate_address_format("localhost:8188").unwrap(),
            "localhost:8188"
        );
    }

    #[test]
    fn rejects_urls_and_metadata_addresses() {
        assert!(validate_address_format("http://127.0.0.1:8188").is_err());
        assert!(validate_address_format("169.254.169.254:80").is_err());
        assert!(validate_address_format("metadata.google.internal:80").is_err());
    }
}
