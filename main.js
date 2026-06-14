const { app, BrowserWindow } = require('electron')
const path = require('path')
const fs = require('fs')
const { spawn } = require('child_process')
const net = require('net')

const pyProcs = []
let rootDir = __dirname

function getRootDir() {
  return app.isPackaged ? process.resourcesPath : __dirname
}

function loadEnv() {
  rootDir = getRootDir()
  const envPath = app.isPackaged
    ? path.join(app.getPath('userData'), '.env')
    : path.join(__dirname, '.env')

  if (app.isPackaged && !fs.existsSync(envPath)) {
    const example = path.join(rootDir, '.env.example')
    if (fs.existsSync(example)) fs.copyFileSync(example, envPath)
  }

  require('dotenv').config({ path: envPath })
}

function portBusy(port) {
  return new Promise((resolve) => {
    const socket = new net.Socket()
    socket.setTimeout(800)
    socket
      .once('connect', () => { socket.destroy(); resolve(true) })
      .once('timeout', () => { socket.destroy(); resolve(false) })
      .once('error', () => resolve(false))
      .connect(port, '127.0.0.1')
  })
}

function findPython() {
  const candidates = [
    path.join(rootDir, '.venv', 'bin', 'python3'),
    path.join(rootDir, '.venv', 'bin', 'python'),
    'python3',
    'python',
  ]
  for (const p of candidates) {
    try {
      if (p.includes('/') && fs.existsSync(p)) return p
    } catch (_) {}
    if (!p.includes('/')) return p
  }
  return 'python3'
}

function startPy(script, label) {
  const python = findPython()
  const proc = spawn(python, [path.join(rootDir, 'python', script)], {
    cwd: path.join(rootDir, 'python'),
    env: { ...process.env },
    stdio: ['ignore', 'pipe', 'pipe'],
  })
  proc.stdout.on('data', (d) => console.log(`[${label}]`, d.toString().trim()))
  proc.stderr.on('data', (d) => console.error(`[${label}]`, d.toString().trim()))
  proc.on('exit', (code) => console.log(`[${label}] stopped (${code})`))
  pyProcs.push(proc)
}

function hasEnv(name) {
  return Boolean((process.env[name] || '').trim())
}

app.whenReady().then(() => {
  loadEnv()
  const calPort = parseInt(process.env.GOOGLE_CALENDAR_PORT || '8768', 10)
  const fcPort = parseInt(process.env.FIRECRAWL_PORT || '8769', 10)

  Promise.all([
    portBusy(8765),
    portBusy(8766),
    portBusy(calPort),
    portBusy(fcPort),
  ]).then(([cvUp, sttUp, calUp, fcUp]) => {
    if (!cvUp) startPy('cv_engine.py', 'cv')
    else console.log('[cv] already running on :8765')

    if (!sttUp) startPy('stt_service.py', 'stt')
    else console.log('[stt] already running on :8766')

    if (hasEnv('GOOGLE_CALENDAR_CLIENT_ID') && hasEnv('GOOGLE_CALENDAR_CLIENT_SECRET')) {
      if (!calUp) startPy('calendar_service.py', 'calendar')
      else console.log(`[calendar] already running on :${calPort}`)
    }

    if (hasEnv('FIRECRAWL_API_KEY')) {
      if (!fcUp) startPy('firecrawl_service.py', 'firecrawl')
      else console.log(`[firecrawl] already running on :${fcPort}`)
    }
  })

  const win = new BrowserWindow({
    width: 1280,
    height: 720,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
    },
  })
  win.loadFile(path.join(rootDir, 'index.html'))
})

app.on('before-quit', () => {
  pyProcs.forEach((p) => {
    try { p.kill() } catch (_) {}
  })
})
