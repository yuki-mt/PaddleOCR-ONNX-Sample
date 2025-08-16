// script.js

// DOM要素の取得
const imageUpload = document.getElementById('imageUpload');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const resultDiv = document.getElementById('result');
const loadingDiv = document.getElementById('loading');

// モデルパスの定義
const MODEL_PATH_DET = 'models/det_model.onnx';
const MODEL_PATH_REC = 'models/rec_model.onnx';
const MODEL_PATH_CLS = 'models/cls_model.onnx';
const DICT_PATH = 'utils/dict.txt';

// グローバ変数
let sessionDet, sessionRec, sessionCls;
let charDict = [];

// 初期化
async function init() {
    console.log('🚀 Starting initialization...');
    try {
        loadingDiv.classList.remove('hidden');
        console.log('🔄 Loading models...');
        // モデルのロード
        await loadModels();
        console.log('✅ Models loaded successfully');
        
        console.log('🔄 Loading dictionary...');
        // 辞書のロード
        await loadDictionary();
        console.log('✅ Dictionary loaded successfully, chars:', charDict.length);
        
        loadingDiv.classList.add('hidden');
        console.log('🎉 Initialization complete!');
    } catch (error) {
        console.error('❌ Error during initialization:', error);
        loadingDiv.textContent = 'Error loading models or dictionary: ' + error.message;
    }
}

// モデルのロード
async function loadModels() {
    console.log('📥 Loading detection model...');
    sessionDet = await ort.InferenceSession.create(MODEL_PATH_DET);
    console.log('✅ Detection model loaded');
    
    console.log('📥 Loading recognition model...');
    sessionRec = await ort.InferenceSession.create(MODEL_PATH_REC);
    console.log('✅ Recognition model loaded');
    
    console.log('📥 Loading classification model...');
    sessionCls = await ort.InferenceSession.create(MODEL_PATH_CLS);
    console.log('✅ Classification model loaded');
}

// 辞書のロード (CTCLabelDecode.add_special_char相当)
async function loadDictionary() {
    console.log('📥 Fetching dictionary file...');
    const response = await fetch(DICT_PATH);
    if (!response.ok) {
        throw new Error(`Failed to fetch dictionary: ${response.status}`);
    }
    const text = await response.text();
    console.log('📄 Dictionary file loaded, length:', text.length);
    
    const lines = text.split('\n').filter(line => line.trim() !== '');
    console.log('📝 Dictionary lines:', lines.length);
    
    // Python版と同じ処理：character_dictから文字リストを作成
    let dictCharacter = [];
    for (const line of lines) {
        const char = line.trim();
        if (char) {
            dictCharacter.push(char);
        }
    }
    console.log('🔤 Characters loaded:', dictCharacter.length);
    
    // use_space_char=True の場合、スペースを追加
    dictCharacter.push(" ");
    
    // CTCLabelDecode.add_special_char相当: 'blank'を先頭に追加
    charDict = ['blank'].concat(dictCharacter);
    console.log('✅ Character dictionary created with', charDict.length, 'characters');
}

// 画像アップロード時の処理
imageUpload.addEventListener('change', async (event) => {
    console.log('📁 File selected');
    const file = event.target.files[0];
    if (!file) {
        console.log('❌ No file selected');
        return;
    }

    console.log('📖 Reading file:', file.name, 'size:', file.size);
    const reader = new FileReader();
    reader.onload = async (e) => {
        console.log('📄 File read complete');
        const img = new Image();
        img.onload = async () => {
            console.log('🖼️ Image loaded:', img.width, 'x', img.height);
            try {
                // Canvasにドロー
                canvas.width = img.width;
                canvas.height = img.height;
                ctx.drawImage(img, 0, 0);
                console.log('🎨 Image drawn on canvas');

                // OCR処理を実行
                console.log('🔍 Starting OCR processing...');
                const results = await processOCR(img);
                console.log('✅ OCR processing complete, results:', results);
                
                // 結果を表示
                displayResults(results);
                console.log('📊 Results displayed');
                
                // ボックスを描画
                drawBoxes(results);
                console.log('📦 Boxes drawn');
            } catch (error) {
                console.error('❌ OCR processing error:', error);
                resultDiv.innerHTML = `<div style="color: red;">Error: ${error.message}</div>`;
            }
        };
        img.onerror = () => {
            console.error('❌ Failed to load image');
        };
        img.src = e.target.result;
    };
    reader.onerror = () => {
        console.error('❌ Failed to read file');
    };
    reader.readAsDataURL(file);
});

// OCR処理 (PaddleOcrONNX.__call__相当)
async function processOCR(image) {
    console.log('🔄 Step 1: Converting image to tensor...');
    // 1. 画像をTensorに変換（検出用前処理）
    const imageTensorInfo = imageToTensor(image);
    console.log('✅ Step 1 complete: Image tensor created', imageTensorInfo.tensor.dims);
    
    console.log('🔄 Step 2: Detecting text...');
    // 2. テキスト検出
    const detResults = await detectText(imageTensorInfo);
    console.log('✅ Step 2 complete: Text detection done, found', detResults.length, 'boxes');
    
    console.log('🔄 Step 3: Sorting boxes...');
    // 3. 検出ボックスをソート
    const sortedBoxes = sortBoxes(detResults);
    console.log('✅ Step 3 complete: Boxes sorted');
    
    console.log('🔄 Step 4: Cropping images from boxes...');
    // 4. 各ボックスから画像を切り出し
    const croppedImages = cropImagesFromBoxes(image, sortedBoxes);
    console.log('✅ Step 4 complete: Cropped', croppedImages.length, 'images');
    
    console.log('🔄 Step 5: Recognizing text...');
    // 5. テキスト認識
    const recResults = await recognizeText(croppedImages);
    console.log('✅ Step 5 complete: Text recognition done', recResults);
    
    console.log('🔄 Step 6: Filtering results...');
    // 6. 結果をフィルタリング
    const filteredResults = filterResults(sortedBoxes, recResults, 0.5); // drop_score = 0.5
    console.log('✅ Step 6 complete: Filtered results', filteredResults);
    
    return filteredResults;
}

// ボックスソート (sorted_boxes相当)
function sortBoxes(boxes) {
    if (!boxes || boxes.length === 0) return [];
    
    // Python版のsorted_boxesに完全に合わせる
    const numBoxes = boxes.length;
    let sortedBoxes = boxes.slice().sort((a, b) => {
        const aY = a.points ? a.points[0][1] : a.y;
        const bY = b.points ? b.points[0][1] : b.y;
        const aX = a.points ? a.points[0][0] : a.x;
        const bX = b.points ? b.points[0][0] : b.x;
        
        // 最初にY座標でソート、次にX座標でソート
        if (Math.abs(aY - bY) < 1e-6) {
            return aX - bX;
        }
        return aY - bY;
    });
    
    // Python版と同じ微調整処理
    for (let i = 0; i < numBoxes - 1; i++) {
        for (let j = i; j >= 0; j--) {
            const currBox = sortedBoxes[j + 1];
            const prevBox = sortedBoxes[j];
            
            const currY = currBox.points ? currBox.points[0][1] : currBox.y;
            const prevY = prevBox.points ? prevBox.points[0][1] : prevBox.y;
            const currX = currBox.points ? currBox.points[0][0] : currBox.x;
            const prevX = prevBox.points ? prevBox.points[0][0] : prevBox.x;
            
            if (Math.abs(currY - prevY) < 10 && currX < prevX) {
                // 位置を交換
                [sortedBoxes[j], sortedBoxes[j + 1]] = [sortedBoxes[j + 1], sortedBoxes[j]];
            } else {
                break;
            }
        }
    }
    
    return sortedBoxes;
}

// 画像の切り出し (get_rotate_crop_image相当)
function cropImagesFromBoxes(image, boxes) {
    const croppedImages = [];
    
    for (const box of boxes) {
        const croppedImage = cropRotateImage(image, box);
        if (croppedImage) {
            croppedImages.push(croppedImage);
        }
    }
    
    return croppedImages;
}

// 回転切り出し (get_rotate_crop_image相当) - Python版に完全対応
function cropRotateImage(image, box) {
    const points = box.points && box.points.length === 4
        ? box.points
        : [
            [box.x, box.y],
            [box.x + box.width, box.y],
            [box.x + box.width, box.y + box.height],
            [box.x, box.y + box.height]
        ];
    
    if (points.length !== 4) {
        console.warn('cropRotateImage: points must be 4*2');
        return null;
    }
    
    // Python版の実装に完全に合わせる
    const imgCropWidth = Math.max(
        Math.sqrt((points[0][0] - points[1][0]) ** 2 + (points[0][1] - points[1][1]) ** 2),
        Math.sqrt((points[2][0] - points[3][0]) ** 2 + (points[2][1] - points[3][1]) ** 2)
    );
    
    const imgCropHeight = Math.max(
        Math.sqrt((points[0][0] - points[3][0]) ** 2 + (points[0][1] - points[3][1]) ** 2),
        Math.sqrt((points[1][0] - points[2][0]) ** 2 + (points[1][1] - points[2][1]) ** 2)
    );
    
    const imgCropWidthInt = Math.round(imgCropWidth);
    const imgCropHeightInt = Math.round(imgCropHeight);
    
    const ptsStd = [
        [0, 0],
        [imgCropWidthInt, 0],
        [imgCropWidthInt, imgCropHeightInt],
        [0, imgCropHeightInt]
    ];
    
    // Canvas上で透視変換を実行
    const canvas = document.createElement('canvas');
    canvas.width = imgCropWidthInt;
    canvas.height = imgCropHeightInt;
    const ctx = canvas.getContext('2d');
    
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    
    // 四角形を2つの三角形に分けて描画
    drawPerspectiveQuad(ctx, image, points, ptsStd);
    
    // Python版と同じ回転判定
    if (imgCropHeightInt * 1.0 / imgCropWidthInt >= 1.5) {
        const rotated = document.createElement('canvas');
        const rctx = rotated.getContext('2d');
        rotated.width = imgCropHeightInt;
        rotated.height = imgCropWidthInt;
        rctx.translate(rotated.width / 2, rotated.height / 2);
        rctx.rotate(Math.PI / 2);
        rctx.drawImage(canvas, -canvas.width / 2, -canvas.height / 2);
        return rotated;
    }
    
    return canvas;
}

// 透視変換の近似
function drawPerspectiveQuad(ctx, image, srcQuad, dstQuad) {
    // 四角形を2つの三角形に分割
    drawTriangle(ctx, image, 
        [srcQuad[0], srcQuad[1], srcQuad[2]], 
        [dstQuad[0], dstQuad[1], dstQuad[2]]
    );
    drawTriangle(ctx, image,
        [srcQuad[0], srcQuad[2], srcQuad[3]],
        [dstQuad[0], dstQuad[2], dstQuad[3]]
    );
}

// 三角形のアフィン変換描画
function drawTriangle(ctx, image, srcTri, dstTri) {
    const matrix = computeAffineFromTriangles(
        { x: srcTri[0][0], y: srcTri[0][1] },
        { x: srcTri[1][0], y: srcTri[1][1] },
        { x: srcTri[2][0], y: srcTri[2][1] },
        { x: dstTri[0][0], y: dstTri[0][1] },
        { x: dstTri[1][0], y: dstTri[1][1] },
        { x: dstTri[2][0], y: dstTri[2][1] }
    );
    
    if (!matrix) return;
    
    ctx.save();
    
    // クリッピングパスを設定
    ctx.beginPath();
    ctx.moveTo(dstTri[0][0], dstTri[0][1]);
    ctx.lineTo(dstTri[1][0], dstTri[1][1]);
    ctx.lineTo(dstTri[2][0], dstTri[2][1]);
    ctx.closePath();
    ctx.clip();
    
    // アフィン変換を適用
    ctx.setTransform(matrix.a, matrix.b, matrix.c, matrix.d, matrix.e, matrix.f);
    ctx.drawImage(image, 0, 0);
    
    ctx.restore();
}

// 3点対応からアフィン行列を求める
function computeAffineFromTriangles(s0, s1, s2, d0, d1, d2) {
    const A = [
        [s0.x, s0.y, 0,    0,    1, 0],
        [0,    0,    s0.x, s0.y, 0, 1],
        [s1.x, s1.y, 0,    0,    1, 0],
        [0,    0,    s1.x, s1.y, 0, 1],
        [s2.x, s2.y, 0,    0,    1, 0],
        [0,    0,    s2.x, s2.y, 0, 1],
    ];
    const B = [d0.x, d0.y, d1.x, d1.y, d2.x, d2.y];

    const params = solve6x6(A, B);
    return { a: params[0], b: params[1], c: params[2], d: params[3], e: params[4], f: params[5] };
}

// 6x6 連立一次方程式を解く
function solve6x6(A, B) {
    const M = A.map((row, i) => [...row, B[i]]);
    const n = 6;
    for (let i = 0; i < n; i++) {
        let maxRow = i;
        for (let r = i + 1; r < n; r++) {
            if (M[r] && Math.abs(M[r][i]) > Math.abs(M[maxRow][i])) maxRow = r;
        }
        if (Math.abs(M[maxRow][i]) < 1e-8) continue;
        [M[i], M[maxRow]] = [M[maxRow], M[i]];
        const pivot = M[i][i];
        for (let c = i; c <= n; c++) M[i][c] /= pivot;
        for (let r = 0; r < n; r++) {
            if (r === i) continue;
            const factor = M[r][i];
            for (let c = i; c <= n; c++) M[r][c] -= factor * M[i][c];
        }
    }
    return M.map(row => row[n]);
}

// テキスト認識 (TextRecognizer.__call__相当)
async function recognizeText(croppedImages) {
    console.log('🔤 Starting text recognition for', croppedImages.length, 'images');
    const results = [];
    const batchSize = 6; // rec_batch_num
    
    // アスペクト比でソート（処理高速化のため）
    console.log('📏 Sorting images by aspect ratio...');
    const imageInfos = croppedImages.map((img, index) => ({
        image: img,
        index: index,
        ratio: img.width / img.height
    }));
    
    imageInfos.sort((a, b) => a.ratio - b.ratio);
    console.log('✅ Images sorted by aspect ratio');
    
    // optional: 角度分類（180度反転の補正）
    if (sessionCls) {
        console.log('🔄 Applying angle classification...');
        for (let i = 0; i < imageInfos.length; i++) {
            const rotated = await maybeRotateByAngleClassifier(imageInfos[i].image);
            if (rotated) {
                console.log('↩️ Image', i, 'rotated 180 degrees');
                imageInfos[i].image = rotated;
            }
        }
        console.log('✅ Angle classification complete');
    }

    // バッチ処理
    console.log('📦 Processing in batches of', batchSize);
    for (let batchStart = 0; batchStart < imageInfos.length; batchStart += batchSize) {
        const batchEnd = Math.min(imageInfos.length, batchStart + batchSize);
        const batch = imageInfos.slice(batchStart, batchEnd);
        console.log(`🔄 Processing batch ${Math.floor(batchStart/batchSize) + 1}: images ${batchStart+1}-${batchEnd}`);
        
        // バッチ内の最大アスペクト比を計算 (Python版と同じ処理)
        const imgC = 3, imgH = 48, imgW = 320;
        let maxRatio = imgW / imgH; // 初期値
        for (const item of batch) {
            maxRatio = Math.max(maxRatio, item.ratio);
        }
        console.log('📏 Max aspect ratio for this batch:', maxRatio);
        
        // バッチ内の画像を前処理
        console.log('🔄 Preprocessing batch images...');
        const normalizedBatch = [];
        for (const item of batch) {
            const normalizedImg = resizeNormImgForRecognition(item.image, maxRatio);
            normalizedBatch.push(normalizedImg);
        }
        console.log('✅ Batch preprocessing complete');
        
        // バッチテンソル作成
        console.log('🔄 Creating batch tensor...');
        const batchTensor = createBatchTensor(normalizedBatch);
        console.log('✅ Batch tensor created:', batchTensor.dims);
        
        // 推論実行
        console.log('🔮 Running recognition inference...');
        const batchResults = await runRecognitionInference(batchTensor);
        console.log('✅ Recognition inference complete, output shape:', batchResults.dims);
        
        // 結果をデコード
        console.log('🔄 Decoding recognition results...');
        const decodedResults = decodeRecognitionResults(batchResults);
        console.log('✅ Batch decoding complete:', decodedResults);
        
        // 結果を元のインデックス順に戻す
        for (let i = 0; i < batch.length; i++) {
            results[batch[i].index] = decodedResults[i];
        }
    }
    
    console.log('🎉 Text recognition complete for all batches');
    return results;
}

// 角度分類を実行し、180度判定なら回転させて返す
async function maybeRotateByAngleClassifier(canvas) {
    try {
        const tensor = preprocessForCls(canvas);
        const feeds = { 'x': tensor };
        const outputs = await sessionCls.run(feeds);
        const out = outputs[Object.keys(outputs)[0]];
        const [b, numClasses] = out.dims;
        if (numClasses < 2) return null;
        // softmax
        const p0 = Math.exp(out.data[0]);
        const p1 = Math.exp(out.data[1]);
        const s = p0 + p1;
        const prob180 = p1 / s;
        if (prob180 >= 0.9) {
            const rotated = document.createElement('canvas');
            const ctx = rotated.getContext('2d');
            rotated.width = canvas.width;
            rotated.height = canvas.height;
            ctx.translate(rotated.width / 2, rotated.height / 2);
            ctx.rotate(Math.PI);
            ctx.drawImage(canvas, -canvas.width / 2, -canvas.height / 2);
            return rotated;
        }
    } catch (e) {
        console.warn('Angle classifier failed:', e);
    }
    return null;
}

function preprocessForCls(canvas) {
    const imgC = 3, imgH = 48, imgW = 192;
    const resize = document.createElement('canvas');
    resize.width = imgW;
    resize.height = imgH;
    const rctx = resize.getContext('2d');
    rctx.drawImage(canvas, 0, 0, imgW, imgH);
    const imgData = rctx.getImageData(0, 0, imgW, imgH).data;
    const out = new Float32Array(imgC * imgH * imgW);
    // CHW, (x/255 - 0.5)/0.5 in BGR order
    for (let h = 0; h < imgH; h++) {
        for (let w = 0; w < imgW; w++) {
            const idx = (h * imgW + w) * 4;
            const r = imgData[idx], g = imgData[idx + 1], b = imgData[idx + 2];
            const bNorm = (b / 255 - 0.5) / 0.5;
            const gNorm = (g / 255 - 0.5) / 0.5;
            const rNorm = (r / 255 - 0.5) / 0.5;
            out[0 * imgH * imgW + h * imgW + w] = bNorm;
            out[1 * imgH * imgW + h * imgW + w] = gNorm;
            out[2 * imgH * imgW + h * imgW + w] = rNorm;
        }
    }
    return new ort.Tensor('float32', out, [1, imgC, imgH, imgW]);
}

// 認識用画像前処理 (TextRecognizer.resize_norm_img相当) - Python版と完全一致
function resizeNormImgForRecognition(canvas, maxRatio) {
    const imgC = 3, imgH = 48, imgW = 320;
    
    // Python版と同じ処理
    const targetImgW = Math.floor(imgH * maxRatio);
    
    const h = canvas.height;
    const w = canvas.width;
    const ratio = w / h;
    
    let resizedW;
    if (Math.ceil(imgH * ratio) > targetImgW) {
        resizedW = targetImgW;
    } else {
        resizedW = Math.ceil(imgH * ratio);
    }
    
    resizedW = Math.min(resizedW, imgW);
    
    // リサイズ
    const resizeCanvas = document.createElement('canvas');
    const resizeCtx = resizeCanvas.getContext('2d');
    resizeCanvas.width = resizedW;
    resizeCanvas.height = imgH;
    
    resizeCtx.drawImage(canvas, 0, 0, resizedW, imgH);
    
    const imageData = resizeCtx.getImageData(0, 0, resizedW, imgH);
    const data = imageData.data;
    
    // Python版と完全に一致する正規化
    // resized_image = resized_image.transpose((2, 0, 1)) / 255
    // resized_image -= 0.5; resized_image /= 0.5
    // padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    // padding_im[:, :, 0:resized_w] = resized_image
    
    const paddingIm = new Float32Array(imgC * imgH * imgW); // ゼロで初期化
    
    for (let c = 0; c < imgC; c++) {
        for (let h = 0; h < imgH; h++) {
            for (let w = 0; w < resizedW; w++) {
                const pixelIndex = (h * resizedW + w) * 4;
                let pixelValue;
                // Python版のOpenCVはBGR順
                if (c === 0) pixelValue = data[pixelIndex + 2]; // B
                else if (c === 1) pixelValue = data[pixelIndex + 1]; // G
                else pixelValue = data[pixelIndex]; // R
                
                // Python版と同じ正規化
                let normalizedValue = pixelValue / 255.0;
                normalizedValue = (normalizedValue - 0.5) / 0.5;
                
                const outputIndex = c * imgH * imgW + h * imgW + w;
                paddingIm[outputIndex] = normalizedValue;
            }
        }
    }
    
    return {
        data: paddingIm,
        shape: [1, imgC, imgH, imgW]
    };
}

// バッチテンソル作成
function createBatchTensor(normalizedBatch) {
    const batchSize = normalizedBatch.length;
    const [, imgC, imgH, imgW] = normalizedBatch[0].shape;
    
    const batchData = new Float32Array(batchSize * imgC * imgH * imgW);
    
    for (let b = 0; b < batchSize; b++) {
        const startIndex = b * imgC * imgH * imgW;
        batchData.set(normalizedBatch[b].data, startIndex);
    }
    
    return new ort.Tensor('float32', batchData, [batchSize, imgC, imgH, imgW]);
}

// 認識推論実行
async function runRecognitionInference(batchTensor) {
    const feeds = { 'x': batchTensor };
    const results = await sessionRec.run(feeds);
    const outputKey = Object.keys(results)[0];
    return results[outputKey];
}

// CTCデコード (Python版CTCLabelDecodeに完全対応)
function decodeRecognitionResults(output) {
    const results = [];
    const dims = output.dims;
    const data = output.data;
    
    if (dims.length !== 3) {
        console.warn('Unexpected recognition output dims:', dims);
        return Array(dims[0] || 0).fill({ text: '', confidence: 0 });
    }

    const B = dims[0]; // batch size
    const T = dims[1]; // sequence length 
    const C = dims[2]; // class number

    // Python版のCTCLabelDecode.__call__に完全対応
    // preds_idx = preds.argmax(axis=2)
    // preds_prob = preds.max(axis=2)
    const predsIdx = new Array(B);
    const predsProb = new Array(B);
    
    for (let b = 0; b < B; b++) {
        predsIdx[b] = new Array(T);
        predsProb[b] = new Array(T);
        
        for (let t = 0; t < T; t++) {
            // 各時刻でsoftmaxを適用
            const logits = new Array(C);
            let maxLogit = -Infinity;
            
            for (let c = 0; c < C; c++) {
                logits[c] = data[b * T * C + t * C + c];
                maxLogit = Math.max(maxLogit, logits[c]);
            }
            
            // Softmax: exp(x_i - max) / sum(exp(x_j - max))
            let sumExp = 0;
            for (let c = 0; c < C; c++) {
                logits[c] = Math.exp(logits[c] - maxLogit);
                sumExp += logits[c];
            }
            
            let maxIdx = 0;
            let maxProb = 0;
            for (let c = 0; c < C; c++) {
                logits[c] /= sumExp;
                if (logits[c] > maxProb) {
                    maxProb = logits[c];
                    maxIdx = c;
                }
            }
            
            predsIdx[b][t] = maxIdx;
            predsProb[b][t] = maxProb;
        }
    }
    
    // CTCLabelDecode.decode (is_remove_duplicate=True)
    for (let b = 0; b < B; b++) {
        const textIndex = predsIdx[b];
        const textProb = predsProb[b];
        
        // selection[1:] = text_index[1:] != text_index[:-1]
        const selection = new Array(T).fill(true);
        for (let i = 1; i < T; i++) {
            selection[i] = textIndex[i] !== textIndex[i - 1];
        }
        
        // ignored_tokens (blank=0) を除外
        for (let i = 0; i < T; i++) {
            if (textIndex[i] === 0) {
                selection[i] = false;
            }
        }
        
        const charList = [];
        const confList = [];
        
        for (let i = 0; i < T; i++) {
            if (selection[i]) {
                const charIdx = textIndex[i];
                if (charIdx > 0 && charIdx < charDict.length) {
                    charList.push(charDict[charIdx]);
                    confList.push(textProb[i]);
                }
            }
        }
        
        const text = charList.join('');
        const confidence = confList.length > 0 ? confList.reduce((a, b) => a + b, 0) / confList.length : 0;
        
        results.push({ text, confidence });
    }
    
    return results;
}

// 結果フィルタリング
function filterResults(boxes, recResults, dropScore) {
    const filteredResults = [];
    
    for (let i = 0; i < Math.min(boxes.length, recResults.length); i++) {
        const box = boxes[i];
        const recResult = recResults[i];
        
        if (recResult && recResult.confidence >= dropScore) {
            filteredResults.push({
                text: recResult.text,
                confidence: recResult.confidence,
                box: box
            });
        }
    }
    
    return filteredResults;
}

// 画像をTensorに変換 (検出用)
function imageToTensor(image) {
    console.log('🖼️ Converting image to tensor, original size:', image.width, 'x', image.height);
    const origWidth = image.width;
    const origHeight = image.height;
    
    // DetResizeForTest相当の処理
    console.log('🔄 Resizing image for detection...');
    const { resizedImage, ratios, destWidth, destHeight } = resizeImageForDetection(image, 960, 'max');
    console.log('📐 Resized to:', destWidth, 'x', destHeight, 'ratios:', ratios);
    
    // 前処理: NormalizeImage相当
    console.log('🔄 Normalizing image...');
    const normalizedData = normalizeImage(
        resizedImage,
        [0.485, 0.456, 0.406], // mean (B,G,R に対応させて使用)
        [0.229, 0.224, 0.225], // std
        1.0 / 255.0            // scale
    );
    console.log('✅ Image normalization complete');
    
    // CHW形式に変換
    console.log('🔄 Converting HWC to CHW format...');
    const chwData = hwc2chw(normalizedData, destHeight, destWidth);
    console.log('✅ CHW conversion complete');
    
    // Tensorを作成 (形状: [1, 3, height, width])
    console.log('🔄 Creating tensor...');
    const tensor = new ort.Tensor('float32', chwData, [1, 3, destHeight, destWidth]);
    console.log('✅ Tensor created:', tensor.dims);
    
    return {
        tensor: tensor,
        shapeInfo: [origHeight, origWidth, ratios[0], ratios[1]]
    };
}

// 検出用画像リサイズ (DetResizeForTest相当)
function resizeImageForDetection(image, limitSideLen, limitType) {
    const origWidth = image.width;
    const origHeight = image.height;
    
    let ratio;
    if (limitType === 'max') {
        ratio = limitSideLen / Math.max(origHeight, origWidth);
    } else if (limitType === 'min') {
        ratio = limitSideLen / Math.min(origHeight, origWidth);
    } else {
        ratio = 1.0;
    }
    
    let resizeH = Math.round(origHeight * ratio);
    let resizeW = Math.round(origWidth * ratio);
    
    // 32の倍数に調整
    resizeH = Math.max(Math.round(resizeH / 32) * 32, 32);
    resizeW = Math.max(Math.round(resizeW / 32) * 32, 32);
    
    // Canvasを作成して画像をリサイズ
    const canvas = document.createElement('canvas');
    canvas.width = resizeW;
    canvas.height = resizeH;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(image, 0, 0, resizeW, resizeH);
    
    const ratioH = resizeH / origHeight;
    const ratioW = resizeW / origWidth;
    
    return {
        resizedImage: canvas,
        ratios: [ratioH, ratioW],
        destWidth: resizeW,
        destHeight: resizeH
    };
}

// 画像正規化 (NormalizeImage相当)
function normalizeImage(canvas, mean, std, scale) {
    const ctx = canvas.getContext('2d');
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    
    const normalizedData = new Float32Array(canvas.width * canvas.height * 3);
    
    for (let i = 0; i < data.length / 4; i++) {
        // BGR順序で正規化
        normalizedData[i * 3] = (data[i * 4 + 2] * scale - mean[0]) / std[0];     // B
        normalizedData[i * 3 + 1] = (data[i * 4 + 1] * scale - mean[1]) / std[1]; // G
        normalizedData[i * 3 + 2] = (data[i * 4] * scale - mean[2]) / std[2];     // R
    }
    
    return normalizedData;
}

// HWC to CHW変換 (ToCHWImage相当)
function hwc2chw(data, height, width) {
    const chwData = new Float32Array(3 * height * width);
    
    for (let c = 0; c < 3; c++) {
        for (let h = 0; h < height; h++) {
            for (let w = 0; w < width; w++) {
                const hwcIdx = h * width * 3 + w * 3 + c;
                const chwIdx = c * height * width + h * width + w;
                chwData[chwIdx] = data[hwcIdx];
            }
        }
    }
    
    return chwData;
}

// テキスト検出 (DBPostProcess相当の後処理を含む)
async function detectText(imageTensorInfo) {
    console.log('🔮 Running detection model inference...');
    const feeds = { 'x': imageTensorInfo.tensor };
    const results = await sessionDet.run(feeds);
    console.log('✅ Detection inference complete');
    
    const output = results['sigmoid_0.tmp_0'] || results[Object.keys(results)[0]];
    console.log('📊 Detection output shape:', output.dims);
    
    const shapeInfo = imageTensorInfo.shapeInfo;
    console.log('🔄 Post-processing detection results...');
    const boxes = postProcessDetection(output, shapeInfo);
    console.log('✅ Detection post-processing complete, found', boxes.length, 'boxes');
    
    return boxes;
}

// 検出結果の後処理 (DBPostProcess相当)
function postProcessDetection(pred, shapeInfo) {
    const [srcH, srcW, ratioH, ratioW] = shapeInfo;
    const thresh = 0.3;
    const boxThresh = 0.6;
    const unclipRatio = 1.5;
    const maxCandidates = 1000;
    
    const batchSize = pred.dims[0];
    const channels = pred.dims[1];
    const height = pred.dims[2];
    const width = pred.dims[3];
    
    const predData = pred.data;
    const segmentation = new Uint8Array(height * width);
    
    // 二値化
    for (let i = 0; i < height * width; i++) {
        segmentation[i] = predData[i] > thresh ? 1 : 0;
    }
    
    const detectedBoxes = findContoursAndCreateBoxes(
        segmentation, width, height, predData,
        boxThresh, unclipRatio, maxCandidates, srcW, srcH
    );

    return detectedBoxes;
}

// 輪郭検出とボックス作成
function findContoursAndCreateBoxes(segmentation, width, height, predData, boxThresh, unclipRatio, maxCandidates, destWidth, destHeight) {
    const boxes = [];
    const visited = new Array(width * height).fill(false);

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const idx = y * width + x;
            if (segmentation[idx] !== 1 || visited[idx]) continue;

            const component = floodFill(segmentation, visited, x, y, width, height);
            if (component.length < 10) continue;

            let scoreSum = 0;
            for (const p of component) {
                scoreSum += predData[p.y * width + p.x];
            }
            const score = scoreSum / component.length;
            if (score < boxThresh) continue;

            const oriented = computeOrientedRect(component);
            if (!oriented) continue;

            const unclipped = unclipRectangle(oriented, unclipRatio);
            if (!unclipped) continue;

            const scaled = scaleBoxToOriginal(unclipped.points, width, height, destWidth, destHeight);
            if (scaled) {
                boxes.push(scaled);
            }
            if (boxes.length >= maxCandidates) break;
        }
    }

    return boxes;
}

// Flood Fill による連結成分検出
function floodFill(segmentation, visited, startX, startY, width, height) {
    const stack = [{x: startX, y: startY}];
    const component = [];
    
    while (stack.length > 0) {
        const {x, y} = stack.pop();
        const idx = y * width + x;
        
        if (x < 0 || x >= width || y < 0 || y >= height || visited[idx] || segmentation[idx] === 0) {
            continue;
        }
        
        visited[idx] = true;
        component.push({x, y});
        
        // 4近傍を追加
        stack.push({x: x + 1, y: y});
        stack.push({x: x - 1, y: y});
        stack.push({x: x, y: y + 1});
        stack.push({x: x, y: y - 1});
    }
    
    return component;
}

// PCA で近似した最小外接矩形を求める
function computeOrientedRect(component) {
    if (!component || component.length < 4) return null;
    
    let sx = 0, sy = 0;
    for (const p of component) { sx += p.x; sy += p.y; }
    const n = component.length;
    const cx = sx / n, cy = sy / n;

    let sxx = 0, syy = 0, sxy = 0;
    for (const p of component) {
        const dx = p.x - cx, dy = p.y - cy;
        sxx += dx * dx; syy += dy * dy; sxy += dx * dy;
    }
    sxx /= n; syy /= n; sxy /= n;

    const angle = 0.5 * Math.atan2(2 * sxy, sxx - syy);
    const cos = Math.cos(angle), sin = Math.sin(angle);

    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const p of component) {
        const rx = cos * (p.x - cx) + sin * (p.y - cy);
        const ry = -sin * (p.x - cx) + cos * (p.y - cy);
        if (rx < minX) minX = rx; if (rx > maxX) maxX = rx;
        if (ry < minY) minY = ry; if (ry > maxY) maxY = ry;
    }
    const w = Math.max(1e-3, maxX - minX);
    const h = Math.max(1e-3, maxY - minY);
    const cxLocal = (minX + maxX) / 2;
    const cyLocal = (minY + maxY) / 2;

    const halfW = w / 2, halfH = h / 2;
    const cornersLocal = [
        [cxLocal - halfW, cyLocal - halfH],
        [cxLocal + halfW, cyLocal - halfH],
        [cxLocal + halfW, cyLocal + halfH],
        [cxLocal - halfW, cyLocal + halfH],
    ];

    const corners = cornersLocal.map(([rx, ry]) => [
        cx + cos * rx - sin * ry,
        cy + sin * rx + cos * ry,
    ]);

    return { points: corners, w, h, cx, cy, cos, sin, cxLocal, cyLocal };
}

// DB の unclip を矩形に対して近似適用
function unclipRectangle(rect, unclipRatio) {
    const { w, h, cx, cy, cos, sin, cxLocal, cyLocal } = rect;
    const area = w * h;
    const perimeter = 2 * (w + h);
    if (perimeter <= 0) return null;
    const d = (area * unclipRatio) / perimeter;

    const w2 = Math.max(1, w + 2 * d);
    const h2 = Math.max(1, h + 2 * d);

    const halfW = w2 / 2, halfH = h2 / 2;
    const cornersLocal = [
        [cxLocal - halfW, cyLocal - halfH],
        [cxLocal + halfW, cyLocal - halfH],
        [cxLocal + halfW, cyLocal + halfH],
        [cxLocal - halfW, cyLocal + halfH],
    ];

    const points = cornersLocal.map(([rx, ry]) => [
        cx + cos * rx - sin * ry,
        cy + sin * rx + cos * ry,
    ]);
    return { points };
}

// ボックス座標を元の画像サイズにスケール
function scaleBoxToOriginal(points, fromWidth, fromHeight, toWidth, toHeight) {
    const scaleX = toWidth / fromWidth;
    const scaleY = toHeight / fromHeight;
    
    const scaledPoints = points.map(([x, y]) => [
        Math.max(0, Math.min(toWidth, Math.round(x * scaleX))),
        Math.max(0, Math.min(toHeight, Math.round(y * scaleY)))
    ]);
    
    const width = Math.hypot(scaledPoints[1][0] - scaledPoints[0][0], scaledPoints[1][1] - scaledPoints[0][1]);
    const height = Math.hypot(scaledPoints[3][0] - scaledPoints[0][0], scaledPoints[3][1] - scaledPoints[0][1]);
    
    if (width <= 3 || height <= 3) return null;
    
    return {
        x: Math.min(...scaledPoints.map(p => p[0])),
        y: Math.min(...scaledPoints.map(p => p[1])),
        width: Math.max(...scaledPoints.map(p => p[0])) - Math.min(...scaledPoints.map(p => p[0])),
        height: Math.max(...scaledPoints.map(p => p[1])) - Math.min(...scaledPoints.map(p => p[1])),
        points: scaledPoints
    };
}

// 結果を表示
function displayResults(results) {
    let resultText = '<h3>OCR Results:</h3><ul>';
    
    for (const result of results) {
        resultText += `<li>${result.text} (Confidence: ${(result.confidence * 100).toFixed(1)}%)</li>`;
    }
    
    resultText += '</ul>';
    resultDiv.innerHTML = resultText;
}

// ボックスを描画
function drawBoxes(results) {
    // Canvas上の既存の画像を保持
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    ctx.putImageData(imageData, 0, 0);
    
    // ボックスを描画
    ctx.strokeStyle = 'red';
    ctx.lineWidth = 2;
    
    for (const result of results) {
        const box = result.box;
        if (box.points && box.points.length === 4) {
            ctx.beginPath();
            ctx.moveTo(box.points[0][0], box.points[0][1]);
            for (let i = 1; i < box.points.length; i++) {
                ctx.lineTo(box.points[i][0], box.points[i][1]);
            }
            ctx.closePath();
            ctx.stroke();
        }
    }
}

// 初期化の実行
init();
