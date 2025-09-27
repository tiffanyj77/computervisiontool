const statusEl = document.getElementById('status');
const fileInput = document.getElementById('file');
const runBtn = document.getElementById('runBtn');
const srcImg = document.getElementById('srcImg');
const srcCanvas = document.getElementById('srcCanvas');
const dstCanvas = document.getElementById('dstCanvas');
const downloadLink = document.getElementById('download');

let cvReady = false, mpReady = false, faceMesh = null;
let blobUrl = null;

// --------- MediaPipe FACE OVAL edges (same as Python used) ----------
const FACEMESH_FACE_OVAL = [
  [10,338],[338,297],[297,332],[332,284],[284,251],[251,389],[389,356],
  [356,454],[454,323],[323,361],[361,288],[288,397],[397,365],[365,379],
  [379,378],[378,400],[400,377],[377,152],[152,148],[148,176],[176,149],
  [149,150],[150,136],[136,172],[172,58],[58,132],[132,93],[93,234],
  [234,127],[127,162],[162,21],[21,54],[54,103],[103,67],[67,109],
  [109,10]  // closed loop
];

// --------- MediaPipe FaceMesh setup ----------
function setupFaceMesh() {
  faceMesh = new FaceMesh({ // remove the second FaceMesh bc FaceMesh is global var
    locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${f}`
  });
  faceMesh.setOptions({
    maxNumFaces: 1,
    refineLandmarks: true,
    selfieMode: false
  });
  faceMesh.onResults(() => {}); // we’ll call .send({image}) and read return via Promise wrapper
  mpReady = true;
  maybeEnable();
}
if (window.FaceMesh) setupFaceMesh();

function maybeEnable() {
  if (cvReady && mpReady) {
    statusEl.textContent = 'OpenCV & FaceMesh ready';
    runBtn.disabled = false;
  }
}

// --------- File -> preview ----------
fileInput.addEventListener('change', () => {
  const f = fileInput.files?.[0];
  if (!f) return;
  if (blobUrl) URL.revokeObjectURL(blobUrl);
  blobUrl = URL.createObjectURL(f);
  srcImg.src = blobUrl;
});

// --------- Main action ----------
runBtn.addEventListener('click', async () => {
  if (!cvReady || !mpReady) return alert('Libraries still loading…');
  const f = fileInput.files?.[0];
  if (!f) return alert('Choose an image first.');

  await waitImg(srcImg);
  drawToCanvas(srcImg, srcCanvas);

  // Start with original image -> dstCanvas
  const {width:w, height:h} = srcCanvas;
  dstCanvas.width = w; dstCanvas.height = h;
  const ctx = dstCanvas.getContext('2d');
  ctx.drawImage(srcCanvas, 0, 0);

  // 1) Guide axes (blue)
  ctx.strokeStyle = '#007bff';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(w/2, 0); ctx.lineTo(w/2, h); // vertical
  ctx.moveTo(0, h/2); ctx.lineTo(w, h/2); // horizontal
  ctx.stroke();

  // 2) Haar cascades for face + eyes (OpenCV.js)
  let src = cv.imread(srcCanvas);
  let gray = new cv.Mat();
  cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);

  // Load cascades into OpenCV FS
  await ensureCascadeLoaded('/assets/haarcascades/haarcascade_frontalface_default.xml', 'face.xml');
  await ensureCascadeLoaded('/assets/haarcascades/haarcascade_eye.xml', 'eye.xml');

  const faceCascade = new cv.CascadeClassifier();
  const eyeCascade  = new cv.CascadeClassifier();
  faceCascade.load('face.xml');
  eyeCascade.load('eye.xml');

  let faces = new cv.RectVector();
  let msize = new cv.Size(0, 0);
  faceCascade.detectMultiScale(gray, faces, 1.1, 8, 0, msize, msize);

  for (let i = 0; i < faces.size(); i++) {
    let r = faces.get(i);
    // ROI
    let roiGray = gray.roi(r);
    // eyes
    let eyes = new cv.RectVector();
    eyeCascade.detectMultiScale(roiGray, eyes, 1.2, 5, 0, msize, msize);
    // collect centers
    let centers = [];
    for (let j = 0; j < eyes.size(); j++) {
      let e = eyes.get(j);
      let cx = e.x + e.width/2;
      let cy = e.y + e.height/2;
      centers.push([cx, cy]);
    }
    if (centers.length >= 2) {
      centers.sort((a,b)=>a[0]-b[0]);
      centers = centers.slice(0,2);
      const eyeY = Math.round((centers[0][1] + centers[1][1]) / 2);
      // draw across the FACE box (blue)
      ctx.beginPath();
      ctx.moveTo(r.x, r.y + eyeY);
      ctx.lineTo(r.x + r.width, r.y + eyeY);
      ctx.stroke();
    }
    eyes.delete(); roiGray.delete();
  }

  // 3) Face outline via MediaPipe FaceMesh
  const landmarks = await runFaceMesh(srcImg); // 468 landmarks (x,y,z in [0..1])
  if (landmarks && landmarks.length) {
    const pts = landmarks.map(lm => [lm.x * w, lm.y * h]);

    // Build ordered polyline for the oval
    const order = orderOval(FACEMESH_FACE_OVAL); // indices in drawing order
    const path = new Path2D();
    const [sx, sy] = pts[order[0]];
    path.moveTo(sx, sy);
    for (let k = 1; k < order.length; k++) {
      const [x, y] = pts[order[k]];
      path.lineTo(x, y);
    }
    path.closePath();

    // draw outline (green)
    ctx.strokeStyle = '#00ff66';
    ctx.lineWidth = 2;
    ctx.stroke(path);
  } else {
    console.log('No face found by FaceMesh.');
  }

  // Cleanup
  faces.delete(); msize.delete(); faceCascade.delete(); eyeCascade.delete();
  gray.delete(); src.delete();

  // Download link
  dstCanvas.toBlob(b => downloadLink.href = URL.createObjectURL(b), 'image/png');
});

// ---------- Utils ----------
function waitImg(img){ return new Promise(res => img.complete ? res() : (img.onload = ()=>res())); }
function drawToCanvas(img, canvas){
  canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
  canvas.getContext('2d').drawImage(img, 0, 0, canvas.width, canvas.height);
}

// Load cascade file into OpenCV’s virtual FS once
const loadedFiles = new Set();
async function ensureCascadeLoaded(url, vfsName){
  if (loadedFiles.has(vfsName)) return;
  const buf = await (await fetch(url)).arrayBuffer();
  cv.FS_createDataFile('/', vfsName, new Uint8Array(buf), true, false);
  loadedFiles.add(vfsName);
}

// Run MediaPipe FaceMesh on a static image element
function runFaceMesh(imageEl){
  return new Promise((resolve) => {
    const fm = new FaceMesh.FaceMesh({
      locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${f}`
    });
    fm.setOptions({maxNumFaces:1, refineLandmarks:true});
    fm.onResults(res => {
      const arr = res.multiFaceLandmarks && res.multiFaceLandmarks[0];
      resolve(arr || null);
    });
    fm.send({image: imageEl});
  });
}

// Convert undirected oval edges into an ordered cycle
function orderOval(edges){
  const nbr = {};
  const idxSet = new Set();
  edges.forEach(([a,b])=>{
    (nbr[a]??=([])).push(b);
    (nbr[b]??=([])).push(a);
    idxSet.add(a); idxSet.add(b);
  });
  // pick leftmost by index heuristic (kept simple; FaceMesh indices are consistent)
  // start from the smallest index in the oval set
  const start = Math.min(...idxSet);
  const order = [start];
  let prev = null, cur = start;
  for (let i=0;i<idxSet.size+5;i++){
    const n = nbr[cur];
    const next = n[0] === prev ? n[1] : n[0];
    if (next === order[0]) break;
    order.push(next);
    prev = cur; cur = next;
  }
  return order;
}
