# VoiceLab 一键安装器 (Windows PowerShell 5.1+)
#
# 用法（普通用户权限即可，全程不写系统目录）:
#   powershell -ExecutionPolicy Bypass -File install.ps1
#   powershell -ExecutionPolicy Bypass -File install.ps1 -InstallDir E:\VoiceLab -Version v0.1.0
#
# 做的事：uv -> 本仓库指定版本 -> 上游 GPT-SoVITS -> 三套 venv -> 全部权重 -> 自检
# 特性：断点续跑（重跑自动跳过已完成步骤）、大文件 Range 续传、GitHub 不通自动走镜像。
param(
    [string]$Version    = "v0.1.0",              # 仓库 tag，也可填 main
    [string]$InstallDir = "$PSScriptRoot\VoiceLab",
    [string]$ZipSource  = ""                     # 镜像前缀，留空=自动（直连失败试 ghproxy.net）
)
$ErrorActionPreference = "Stop"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$ProgressPreference = "SilentlyContinue"

$Repo     = "Jinky-0609/VoiceLab"
$Upstream = "RVC-Boss/GPT-SoVITS"
$script:Step = 0

function Say($m)  { Write-Host "[*] $m" -ForegroundColor Cyan }
function Ok($m)   { Write-Host "[OK] $m" -ForegroundColor Green }
function Warn($m) { Write-Host "[!!] $m" -ForegroundColor DarkYellow }
function Head($m) { $script:Step++; Write-Host "`n===== $script:Step/7  $m =====" -ForegroundColor Yellow }
# PS 5.1 陷阱：EAP=Stop 时原生命令的 stderr（uv 进度、python traceback）会变成
# error record 终止脚本，2>$null 挡不住。所有 native 调用一律走 Run。
function Run([scriptblock]$sb) {
    $eap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { & $sb } finally { $ErrorActionPreference = $eap }
}
function Quiet([scriptblock]$sb) {
    $eap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { & $sb 2>&1 | Out-Null } finally { $ErrorActionPreference = $eap }
}

function Die($m)  {
    Write-Host "[FAIL] $m" -ForegroundColor Red
    Write-Host "已完成的步骤会被跳过——修好网络/权限后直接重跑本脚本即可。"
    exit 1
}

# ---------- 下载：镜像回退 + Range 续传 + 完整性校验 ----------
# minBytes 只用来挡"错误页伪装成文件"（HTML 一般 <100KB），**不要**按预期体积卡上限：
# 上游 GPT-SoVITS 的 zip 实测仅 6.88MB，若照 git clone 的 43MB 设 10MB 门槛，
# 正常下载会被误判失败（本脚本踩过）。真校验交给 Test-Archive 的魔数。
function Get-Url($url, $out, [long]$minBytes = 1) {
    $list = New-Object System.Collections.Generic.List[string]
    if ($ZipSource) { $list.Add("$ZipSource$url") }
    $list.Add($url)
    if (-not $ZipSource -and $url -match "^https://(github|codeload|raw)\.github") {
        $list.Add("https://ghproxy.net/$url")
    }
    foreach ($u in $list) {
        for ($try = 1; $try -le 3; $try++) {
            try {
                $have = 0
                if (Test-Path $out) { $have = (Get-Item $out).Length }
                if ($have -ge $minBytes) { return $true }
                $curlArgs = @("--ssl-no-revoke", "-sS", "-L", "--fail", "--max-time", "3600",
                              "--retry", "2", "--create-dirs", "-o", $out)
                if ($have -gt 0) { $curlArgs += @("-C", "-") }
                $curlArgs += $u
                Run { & curl.exe @curlArgs 2>&1 | Out-Null }
                if ($LASTEXITCODE -eq 0 -or $LASTEXITCODE -eq 33) {
                    if ((Test-Path $out) -and ((Get-Item $out).Length -ge $minBytes)) { return $true }
                }
            } catch { }
            Start-Sleep ([Math]::Min(3 * $try, 10))
        }
        Warn "该源未成功，换下一个：$($u.Substring(0, [Math]::Min(72, $u.Length)))"
    }
    return $false
}

# zip/tar.gz 魔数校验：404 的 HTML 也会"下载成功"，必须验
function Test-Archive($path) {
    if (-not (Test-Path $path)) { return $false }
    $fs = [IO.File]::OpenRead($path)
    try { $b = New-Object byte[] 4; [void]$fs.Read($b, 0, 4) } finally { $fs.Close() }
    $isZip = ($b[0] -eq 0x50 -and $b[1] -eq 0x4B)
    $isGz  = ($b[0] -eq 0x1F -and $b[1] -eq 0x8B)
    return ($isZip -or $isGz)
}

Say "VoiceLab 安装器 | 版本 $Version | 目标 $InstallDir"
if (-not (Get-Command curl.exe -ErrorAction SilentlyContinue)) { Die "缺少 curl.exe（Win10 1803+ 自带）" }
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null

# ---------- 1/7 uv ----------
Head "检查 Python 包管理器 uv"
$uv = $null
$c = Get-Command uv -ErrorAction SilentlyContinue
if ($c) { $uv = $c.Source }
if (-not $uv) {
    $cand = @(
        "$env:USERPROFILE\.local\bin\uv.exe",
        "$env:LOCALAPPDATA\Programs\uv\uv.exe"
    )
    foreach ($p in $cand) { if (Test-Path $p) { $uv = $p; break } }
}
if (-not $uv) {
    $found = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Filter uv.exe -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($found) { $uv = $found.FullName }
}
if ($uv) { Ok "uv: $uv" } else {
    Say "安装 uv（官方脚本）..."
    $installer = Join-Path $env:TEMP "uv-install.ps1"
    if (-not (Get-Url "https://astral.sh/uv/install.ps1" $installer 1000)) { Die "uv 安装脚本下载失败，请检查网络" }
    Run { & powershell -ExecutionPolicy Bypass -File $installer 2>&1 | Out-Null }
    $uv = Join-Path $env:USERPROFILE ".local\bin\uv.exe"
    if (-not (Test-Path $uv)) { Die "uv 安装失败（或需重开终端后重跑本脚本）" }
    Ok "uv 已装到 $uv"
}

# ---------- 2/7 本仓库代码 ----------
Head "下载 VoiceLab $Version"
$mark = Join-Path $InstallDir ".installed_version"
if ((Test-Path $mark) -and ((Get-Content $mark -Raw).Trim() -eq $Version)) {
    Ok "代码已是 $Version（跳过）"
} else {
    if ($Version -eq "main") {
        $url = "https://codeload.github.com/$Repo/tar.gz/refs/heads/main"
        $zip = Join-Path $env:TEMP "voicelab-$Version.tar.gz"
    } else {
        $url = "https://codeload.github.com/$Repo/archive/refs/tags/$Version.zip"
        $zip = Join-Path $env:TEMP "voicelab-$Version.zip"
    }
    Remove-Item $zip -Force -ErrorAction SilentlyContinue
    if (-not (Get-Url $url $zip 100KB)) { Die "仓库下载失败（已试直连与镜像）。可指定 -Version main 或 -ZipSource" }
    if (-not (Test-Archive $zip)) { Die "下载内容不是有效压缩包（tag $Version 可能不存在）" }
    $tmp = Join-Path $env:TEMP "voicelab-extract"
    Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Path $tmp | Out-Null
    if ($url.EndsWith(".zip")) { Expand-Archive $zip -DestinationPath $tmp }
    else                       { Run { & tar.exe -xzf $zip -C $tmp 2>&1 | Out-Null }; if ($LASTEXITCODE -ne 0) { Die "tar 解压失败" } }
    $src = Get-ChildItem $tmp -Directory | Select-Object -First 1
    if (-not $src) { Die "解压后未找到内容" }
    Copy-Item -Path (Join-Path $src.FullName "*") -Destination $InstallDir -Recurse -Force
    Set-Content -Path $mark -Value $Version -NoNewline
    Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
    Ok "代码 -> $InstallDir"
}

# ---------- 3/7 上游 GPT-SoVITS ----------
Head "下载上游 GPT-SoVITS 代码（git 仓库不含权重，约 43MB）"
$up = Join-Path $InstallDir "_src\GPT-SoVITS-main"
if (Test-Path (Join-Path $up "GPT_SoVITS")) { Ok "上游已就位（跳过）" } else {
    $uzip = Join-Path $env:TEMP "gpt-sovits.zip"
    Remove-Item $uzip -Force -ErrorAction SilentlyContinue
    if (-not (Get-Url "https://codeload.github.com/$Upstream/zip/refs/heads/main" $uzip 1MB)) { Die "上游下载失败" }
    if (-not (Test-Archive $uzip)) { Die "上游下载内容不是有效压缩包" }
    $utmp = Join-Path $env:TEMP "gpt-sovits-extract"
    Remove-Item $utmp -Recurse -Force -ErrorAction SilentlyContinue
    Expand-Archive $uzip -DestinationPath $utmp
    $usrc = Get-ChildItem $utmp -Directory | Select-Object -First 1
    New-Item -ItemType Directory -Force -Path (Join-Path $InstallDir "_src") | Out-Null
    Move-Item $usrc.FullName $up
    Remove-Item $utmp -Recurse -Force -ErrorAction SilentlyContinue
    Ok "上游 -> $up"
}

# ---------- 4/7 三套 venv ----------
Head "创建虚拟环境并安装依赖（最大项：含 PyTorch CUDA 版，下载 6~9GB）"
Push-Location $InstallDir
$envs = @(
    @{ n = ".venv-tts"; r = "requirements/tts.txt"; extra = $true;  probe = "import torch" },
    @{ n = ".venv-asr"; r = "requirements/asr.txt"; extra = $true;  probe = "import qwen_asr" },
    @{ n = ".venv-uvr"; r = "requirements/uvr.txt"; extra = $false; probe = "import audio_separator" }
)
foreach ($e in $envs) {
    if (-not (Test-Path (Join-Path $e.n "pyvenv.cfg"))) {
        Run { & $uv venv $e.n --python 3.12 2>&1 | Out-Null }
        Ok "创建 $($e.n)"
    }
    $py = Join-Path $e.n "Scripts\python.exe"
    Quiet { & $py -c $e.probe }
    if ($LASTEXITCODE -eq 0) { Ok "$($e.n) 依赖已装（跳过）"; continue }
    Say "安装 $($e.r) ..."
    $ua = @("pip", "install", "-p", $py, "-r", $e.r)
    if ($e.extra) { $ua += @("--extra-index-url", "https://download.pytorch.org/whl/cu128", "--index-strategy", "unsafe-best-match") }
    Run { & $uv @ua 2>&1 | ForEach-Object { Write-Host $_ } }
    if ($LASTEXITCODE -ne 0) {
        Die "$($e.n) 依赖安装失败。若报缓存锁超时：删 %LOCALAPPDATA%\uv\cache\.lock 后重跑；或加 --no-cache 手动装"
    }
    Ok "$($e.n) 完成"
}

# ---------- 5/7 权重 ----------
Head "下载模型权重（GPT-SoVITS 约 3.3GB + Qwen3-ASR 约 1.8GB，均断点续传）"
Run { & ".venv-tts\Scripts\python.exe" "tools\fetch_models.py" 2>&1 | ForEach-Object { Write-Host $_ } }
if ($LASTEXITCODE -ne 0) { Die "TTS 权重不齐（重跑本脚本会续传）" }
Run { & ".venv-asr\Scripts\python.exe" "tools\fetch_qwen_asr.py" 2>&1 | ForEach-Object { Write-Host $_ } }
if ($LASTEXITCODE -ne 0) { Die "ASR 权重不齐（重跑本脚本会续传）" }
Ok "权重就绪"

# ---------- 6/7 自检 ----------
Head "完整性自检"
Run { & ".venv-tts\Scripts\python.exe" "tools\test_guard.py" 2>&1 | ForEach-Object { Write-Host $_ } }
if ($LASTEXITCODE -ne 0) { Die "守卫自检失败" }
Quiet { & ".venv-tts\Scripts\python.exe" "tools\fetch_models.py" --check }
if ($LASTEXITCODE -ne 0) { Die "权重体检失败" }
Ok "全部检查通过"

# ---------- 7/7 收尾 ----------
Head "准备配置与使用说明"
if (-not (Test-Path "config.json")) {
    Copy-Item "config.example.json" "config.json"
    Say "已生成 config.json（当前只含 zeroshot 骨架）"
}
Write-Host ""
Say "放一段参考音频：data\refs\my.wav + 逐字文本 data\refs\my.txt（3~10 秒、单人、干声）"
Say "命令行试听："
Write-Host '  .venv-tts\Scripts\python.exe tools\voicelab.py speak -p zeroshot -r data\refs\my.wav -R data\refs\my.txt --ref-lang 中文 -t "你好，测试。" -l 中文 -o out\test.wav' -ForegroundColor White
Say "图形界面（已装则）："
Write-Host '  .venv-tts\Scripts\python.exe tools\webui.py' -ForegroundColor White
Say "文档：README.md、docs\环境重建.md ｜ 反馈：https://github.com/$Repo/issues"
Pop-Location
Write-Host "`n########## 安装完成 ##########" -ForegroundColor Green
