$ErrorActionPreference = "Stop"

$Repo = "sunerpy/bedrock-gateway-rust"
$Bin = "bedrock-gateway"
$ChecksumFile = "SHA256SUMS"

function Die($Message) {
  Write-Error $Message
  exit 1
}

switch ($env:PROCESSOR_ARCHITECTURE) {
  "AMD64" { $Arch = "x86_64" }
  "ARM64" { $Arch = "aarch64" }
  default { Die "unsupported architecture: $env:PROCESSOR_ARCHITECTURE" }
}

# Releases up to 0.18.0 are tagged bedrock-gateway-rust-vX.Y.Z, later ones vX.Y.Z.
$LegacyTagPrefix = "bedrock-gateway-rust-"

if ($env:TOOL_VERSION) {
  $Version = $env:TOOL_VERSION -replace '^v', ''
  $Tag = "v$Version"
  try {
    Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/releases/tags/$Tag" `
      -Headers @{ "User-Agent" = "install-script" } | Out-Null
  } catch {
    $Tag = "${LegacyTagPrefix}v$Version"
  }
} else {
  $Release = Invoke-RestMethod `
    -Uri "https://api.github.com/repos/$Repo/releases/latest" `
    -Headers @{ "User-Agent" = "install-script" }
  $Tag = $Release.tag_name
  $Version = $Tag -replace '^.*v(?=[0-9])', ''
}

$Target = "$Arch-pc-windows-msvc"
$Asset = "$Bin-$Version-$Target.zip"
$BaseUrl = "https://github.com/$Repo/releases/download/$Tag"
$InstallDir = if ($env:TOOL_INSTALL_DIR) { $env:TOOL_INSTALL_DIR } else { "$HOME\.local\bin" }
$TempDir = New-Item -ItemType Directory -Path (Join-Path $env:TEMP ([System.Guid]::NewGuid()))

try {
  $Archive = Join-Path $TempDir $Asset
  $Checksums = Join-Path $TempDir $ChecksumFile
  Invoke-WebRequest -Uri "$BaseUrl/$Asset" -OutFile $Archive
  Invoke-WebRequest -Uri "$BaseUrl/$ChecksumFile" -OutFile $Checksums

  $EscapedAsset = [Regex]::Escape($Asset)
  $Line = Get-Content $Checksums | Where-Object { $_ -match "\s\*?$EscapedAsset$" } | Select-Object -First 1
  if (-not $Line) { Die "$Asset is missing from $ChecksumFile" }

  $Expected = ($Line -split '\s+')[0].ToLowerInvariant()
  $Actual = (Get-FileHash -Algorithm SHA256 -Path $Archive).Hash.ToLowerInvariant()
  if ($Actual -ne $Expected) { Die "checksum mismatch for $Asset" }

  Expand-Archive -Path $Archive -DestinationPath $TempDir -Force
  New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
  Move-Item -Force -Path (Join-Path $TempDir "$Bin.exe") -Destination (Join-Path $InstallDir "$Bin.exe")
  Write-Host "Installed $Bin to $InstallDir\$Bin.exe"
} finally {
  Remove-Item -Recurse -Force $TempDir -ErrorAction SilentlyContinue
}
