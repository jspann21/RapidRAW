const SERVICE_NAME: &str = "io.github.CyberTimon.RapidRAW";

#[cfg(any(target_os = "windows", target_os = "macos", target_os = "linux"))]
fn entry(key: &str) -> Result<keyring::Entry, String> {
    keyring::Entry::new(SERVICE_NAME, key)
        .map_err(|e| format!("Failed to open secure credential store: {}", e))
}

#[cfg(any(target_os = "windows", target_os = "macos", target_os = "linux"))]
pub fn get_secret(key: &str) -> Result<Option<String>, String> {
    match entry(key)?.get_password() {
        Ok(secret) => Ok(Some(secret)),
        Err(keyring::Error::NoEntry) => Ok(None),
        Err(e) => Err(format!("Failed to read secure credential: {}", e)),
    }
}

#[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
pub fn get_secret(_key: &str) -> Result<Option<String>, String> {
    Ok(None)
}

#[cfg(any(target_os = "windows", target_os = "macos", target_os = "linux"))]
pub fn set_secret(key: &str, value: &str) -> Result<(), String> {
    entry(key)?
        .set_password(value)
        .map_err(|e| format!("Failed to write secure credential: {}", e))
}

#[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
pub fn set_secret(_key: &str, _value: &str) -> Result<(), String> {
    Err("Secure credential storage is not available on this platform.".to_string())
}

#[cfg(any(target_os = "windows", target_os = "macos", target_os = "linux"))]
pub fn delete_secret(key: &str) -> Result<(), String> {
    match entry(key)?.delete_credential() {
        Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
        Err(e) => Err(format!("Failed to delete secure credential: {}", e)),
    }
}

#[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
pub fn delete_secret(_key: &str) -> Result<(), String> {
    Ok(())
}
