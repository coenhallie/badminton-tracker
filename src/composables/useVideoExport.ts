import { ref, type Ref } from 'vue'

export interface UseVideoExportOptions {
  videoRef: Ref<HTMLVideoElement | null>
  canvasRef: Ref<HTMLCanvasElement | null>
  findFrameAtOrBeforeFn: (time: number) => unknown
  showHeatmap: Ref<boolean>
}

export function useVideoExport({
  videoRef,
  canvasRef,
  findFrameAtOrBeforeFn,
  showHeatmap,
}: UseVideoExportOptions) {
  const isExporting = ref(false)
  const exportProgress = ref(0)

  let mediaRecorder: MediaRecorder | null = null
  let recordedChunks: Blob[] = []
  let savedTime = 0
  let savedRate = 1
  let savedMuted = false
  let cancelled = false

  function getSupportedMimeType(): string {
    const candidates = [
      'video/webm;codecs=vp9,opus',
      'video/webm;codecs=vp9',
      'video/webm;codecs=vp8,opus',
      'video/webm;codecs=vp8',
      'video/webm',
    ]
    for (const mime of candidates) {
      if (MediaRecorder.isTypeSupported(mime)) return mime
    }
    return ''
  }

  function startExport() {
    const video = videoRef.value
    const overlayCanvas = canvasRef.value
    if (!video || !overlayCanvas) return

    cancelled = false
    isExporting.value = true
    exportProgress.value = 0
    recordedChunks = []

    // Save current state
    savedTime = video.currentTime
    savedRate = video.playbackRate
    savedMuted = video.muted

    // Create offscreen compositing canvas
    const compCanvas = document.createElement('canvas')
    compCanvas.width = video.videoWidth
    compCanvas.height = video.videoHeight
    const compCtx = compCanvas.getContext('2d')!

    // Build the recording stream
    const canvasStream = compCanvas.captureStream(30)

    // Try to capture audio from the video element
    try {
      const videoStream = (video as HTMLVideoElement & { captureStream(): MediaStream }).captureStream()
      for (const track of videoStream.getAudioTracks()) {
        canvasStream.addTrack(track)
      }
    } catch {
      // Audio capture not supported or tainted — proceed without audio
    }

    const mimeType = getSupportedMimeType()
    const options: MediaRecorderOptions = { videoBitsPerSecond: 8_000_000 }
    if (mimeType) options.mimeType = mimeType

    mediaRecorder = new MediaRecorder(canvasStream, options)

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) recordedChunks.push(e.data)
    }

    mediaRecorder.onstop = () => {
      if (cancelled) return
      // Build and download
      const blob = new Blob(recordedChunks, { type: mimeType || 'video/webm' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `shuttl-export-${Date.now()}.webm`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)

      isExporting.value = false
      exportProgress.value = 100
    }

    // Start recording with 1s timeslice
    mediaRecorder.start(1000)

    // Prepare video for playthrough
    video.currentTime = 0
    video.playbackRate = 1
    video.muted = true

    // Wait for seek to complete, then play
    const onSeeked = () => {
      video.removeEventListener('seeked', onSeeked)
      video.play()
      requestCompositeFrame()
    }
    video.addEventListener('seeked', onSeeked)

    // When video reaches the end, finish recording
    const onEnded = () => {
      video.removeEventListener('ended', onEnded)
      if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop()
      }
      restoreVideoState()
    }
    video.addEventListener('ended', onEnded, { once: true })

    function requestCompositeFrame() {
      if (cancelled || !video) return

      // Prefer requestVideoFrameCallback for frame-accurate compositing.
      // Guarded via runtime lookup so older browsers fall back to rAF —
      // lib.dom's type declares rVFC as always-present, which would
      // otherwise make TS narrow the else branch to never.
      const rvfc = (video as HTMLVideoElement).requestVideoFrameCallback?.bind(video)
      if (rvfc) {
        rvfc((_now, metadata) => {
          if (cancelled) return
          compositeFrame(metadata.mediaTime)
          requestCompositeFrame()
        })
      } else {
        // Fallback: use requestAnimationFrame
        const v = video
        requestAnimationFrame(() => {
          if (cancelled) return
          compositeFrame(v.currentTime)
          requestCompositeFrame()
        })
      }
    }

    function compositeFrame(currentTime: number) {
      if (!video || !compCtx || !overlayCanvas) return

      // Update progress
      if (video.duration && isFinite(video.duration)) {
        exportProgress.value = Math.min(99, Math.round((currentTime / video.duration) * 100))
      }

      // Draw video frame (with brightness filter if heatmap is active)
      if (showHeatmap.value) {
        compCtx.filter = 'brightness(0.4)'
      } else {
        compCtx.filter = 'none'
      }
      compCtx.drawImage(video, 0, 0, compCanvas.width, compCanvas.height)
      compCtx.filter = 'none'

      // Trigger overlay redraw so the overlay canvas is current
      findFrameAtOrBeforeFn(currentTime)

      // Draw overlay canvas on top
      compCtx.drawImage(overlayCanvas, 0, 0, compCanvas.width, compCanvas.height)
    }
  }

  function restoreVideoState() {
    const video = videoRef.value
    if (!video) return
    video.pause()
    video.currentTime = savedTime
    video.playbackRate = savedRate
    video.muted = savedMuted
  }

  function cancelExport() {
    cancelled = true
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
      mediaRecorder.stop()
    }
    restoreVideoState()
    isExporting.value = false
    exportProgress.value = 0
    recordedChunks = []
  }

  return {
    isExporting,
    exportProgress,
    startExport,
    cancelExport,
  }
}
