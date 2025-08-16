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
    try {
        loadingDiv.classList.remove('hidden');
        // モデルのロード
        await loadModels();
        // 辞書のロード
        await loadDictionary();
        loadingDiv.classList.add('hidden');
        console.log('Models and dictionary loaded successfully');
    } catch (error) {
        console.error('Error during initialization:', error);
        loadingDiv.textContent = 'Error loading models or dictionary: ' + error.message;
    }
}

// モデルのロード
async function loadModels() {
    sessionDet = await ort.InferenceSession.create(MODEL_PATH_DET);
    sessionRec = await ort.InferenceSession.create(MODEL_PATH_REC);
    sessionCls = await ort.InferenceSession.create(MODEL_PATH_CLS);
}

// 辞書のロード (CTCLabelDecode.add_special_char相当)
async function loadDictionary() {
    const response = await fetch(DICT_PATH);
    const text = await response.text();
    const lines = text.split('\n').filter(line => line.trim() !== '');
    
    // Python版と同じ処理：character_dictから文字リストを作成
    let dictCharacter = [];
    for (const line of lines) {
        const char = line.trim();
        if (char) {
            dictCharacter.push(char);
        }
    }
    
    // use_space_char=True の場合、スペースを追加
    dictCharacter.push(" ");
    
    // CTCLabelDecode.add_special_char相当: 'blank'を先頭に追加
    charDict = ['blank'].concat(dictCharacter);
}

// 画像アップロード時の処理
imageUpload.addEventListener('change', async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = async (e) => {
        const img = new Image();
        img.onload = async () => {
            try {
                loadingDiv.classList.remove('hidden');
                loadingDiv.textContent = 'Processing OCR...';
                
                // Canvasに画像を描画
                canvas.width = img.width;
                canvas.height = img.height;
                ctx.drawImage(img, 0, 0);
                
                // OCR処理
                const ocrResults = await processOCR(img);
                
                // 結果の表示
                displayResults(ocrResults);
                
                canvas.classList.remove('hidden');
                resultDiv.classList.remove('hidden');
                loadingDiv.classList.add('hidden');
            } catch (error) {
                console.error('Error processing image:', error);
                loadingDiv.textContent = 'Error processing image: ' + error.message;
            }
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
});

// OCR処理 (PaddleOcrONNX.__call__相当)
async function processOCR(image) {
    // 1. 画像をTensorに変換（検出用前処理）
    const imageTensorInfo = imageToTensor(image);
    
    // 2. テキスト検出
    const detResults = await detectText(imageTensorInfo);
    
    // 3. 検出ボックスをソート
    const sortedBoxes = sortBoxes(detResults);
    
    // 4. 各ボックスから画像を切り出し
    const croppedImages = cropImagesFromBoxes(image, sortedBoxes);
    
    // 5. テキスト認識
    const recResults = await recognizeText(croppedImages);
    
    // 6. 結果をフィルタリング
    const filteredResults = filterResults(sortedBoxes, recResults, 0.5); // drop_score = 0.5
    
    return filteredResults;
}

// ボックスソート (sorted_boxes相当)
function sortBoxes(boxes) {
    if (!boxes || boxes.length === 0) return [];
    
    // 上から下、左から右の順にソート
    let sortedBoxes = boxes.slice().sort((a, b) => {
        const aY = a.points ? a.points[0][1] : a.y;
        const bY = b.points ? b.points[0][1] : b.y;
        const aX = a.points ? a.points[0][0] : a.x;
        const bX = b.points ? b.points[0][0] : b.x;
        
        // Y座標の差が10px以内なら同じ行とみなしてX座標でソート
        if (Math.abs(aY - bY) < 10) {
            return aX - bX;
        }
        return aY - bY;
    });
    
    // 同じ行内での細かいソート調整
    for (let i = 0; i < sortedBoxes.length - 1; i++) {
        for (let j = i; j >= 0; j--) {
            const currY = sortedBoxes[j + 1].points ? sortedBoxes[j + 1].points[0][1] : sortedBoxes[j + 1].y;
            const prevY = sortedBoxes[j].points ? sortedBoxes[j].points[0][1] : sortedBoxes[j].y;
            const currX = sortedBoxes[j + 1].points ? sortedBoxes[j + 1].points[0][0] : sortedBoxes[j + 1].x;
            const prevX = sortedBoxes[j].points ? sortedBoxes[j].points[0][0] : sortedBoxes[j].x;
            
            if (Math.abs(currY - prevY) < 10 && currX < prevX) {
                // 交換
                const temp = sortedBoxes[j];
                sortedBoxes[j] = sortedBoxes[j + 1];
                sortedBoxes[j + 1] = temp;
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

// 回転切り出し (get_rotate_crop_image相当)
function cropRotateImage(image, box) {
    // Python版の get_rotate_crop_image に完全に合わせる
    const points = box.points && box.points.length === 4
        ? box.points
        : [
            [box.x, box.y],
            [box.x + box.width, box.y],
            [box.x + box.width, box.y + box.height],
            [box.x, box.y + box.height]
        ];
    
    // assert len(points) == 4, "shape of points must be 4*2"
    if (points.length !== 4) {
        console.warn('cropRotateImage: points must be 4*2');
        return null;
    }
    
    // img_crop_width = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
    const imgCropWidth = Math.max(
        Math.sqrt((points[0][0] - points[1][0]) ** 2 + (points[0][1] - points[1][1]) ** 2),
        Math.sqrt((points[2][0] - points[3][0]) ** 2 + (points[2][1] - points[3][1]) ** 2)
    );
    
    // img_crop_height = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))
    const imgCropHeight = Math.max(
        Math.sqrt((points[0][0] - points[3][0]) ** 2 + (points[0][1] - points[3][1]) ** 2),
        Math.sqrt((points[1][0] - points[2][0]) ** 2 + (points[1][1] - points[2][1]) ** 2)
    );
    
    const imgCropWidthInt = Math.round(imgCropWidth);
    const imgCropHeightInt = Math.round(imgCropHeight);
    
    // pts_std = np.float32([[0, 0], [img_crop_width, 0], [img_crop_width, img_crop_height], [0, img_crop_height]])
    const ptsStd = [
        [0, 0],
        [imgCropWidthInt, 0],
        [imgCropWidthInt, imgCropHeightInt],
        [0, imgCropHeightInt]
    ];
    
    // OpenCVのgetPerspectiveTransformとwarpPerspectiveに相当する処理
    const srcPoints = points.map(p => [p[0], p[1]]);
    
    // Canvas上で透視変換を実行
    const canvas = document.createElement('canvas');
    canvas.width = imgCropWidthInt;
    canvas.height = imgCropHeightInt;
    const ctx = canvas.getContext('2d');
    
    // 透視変換行列を計算してsetTransformを使用
    const matrix = computePerspectiveTransform(srcPoints, ptsStd);
    if (!matrix) {
        console.warn('Failed to compute perspective transform');
        return null;
    }
    
    // Canvas 2Dでは真の透視変換はできないので、アフィン変換で近似
    // より正確な実装のため、四角形を3つの三角形に分けて描画
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    
    // 透視変換の近似として、四角形を2つの三角形に分けて描画
    drawPerspectiveQuad(ctx, image, srcPoints, ptsStd);
    
    // dst_img_height, dst_img_width = dst_img.shape[0:2]
    // if dst_img_height * 1.0 / dst_img_width >= 1.5: dst_img = np.rot90(dst_img)
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

// 透視変換の近似（四角形を三角形に分割して描画）
function drawPerspectiveQuad(ctx, image, srcQuad, dstQuad) {
    // 四角形を2つの三角形に分割
    // 三角形1: 0-1-2
    drawTriangle(ctx, image, 
        [srcQuad[0], srcQuad[1], srcQuad[2]], 
        [dstQuad[0], dstQuad[1], dstQuad[2]]
    );
    // 三角形2: 0-2-3  
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

// 透視変換行列の計算（実際は使用しない）
function computePerspectiveTransform(src, dst) {
    // 簡易版：アフィン変換で近似
    return computeAffineFromTriangles(
        { x: src[0][0], y: src[0][1] },
        { x: src[1][0], y: src[1][1] },
        { x: src[3][0], y: src[3][1] },
        { x: dst[0][0], y: dst[0][1] },
        { x: dst[1][0], y: dst[1][1] },
        { x: dst[3][0], y: dst[3][1] }
    );
}

// 2点間距離
function distance(a, b) {
    const dx = a[0] - b[0];
    const dy = a[1] - b[1];
    return Math.hypot(dx, dy);
}

// 4点を TL, TR, BR, BL の順に並べ替える
function orderQuadPoints(points) {
    const pts = points.map(p => ({ x: p[0], y: p[1] }));
    // 左上: x+y が最小, 右下: x+y が最大
    const sumSorted = [...pts].sort((p1, p2) => (p1.x + p1.y) - (p2.x + p2.y));
    const tl = sumSorted[0];
    const br = sumSorted[3];
    // 右上: x-y が最大, 左下: x-y が最小
    const diffSorted = [...pts].sort((p1, p2) => (p1.x - p1.y) - (p2.x - p2.y));
    const bl = diffSorted[0];
    const tr = diffSorted[3];
    return [ [tl.x, tl.y], [tr.x, tr.y], [br.x, br.y], [bl.x, bl.y] ];
}

// 3点対応からアフィン行列（a,b,c,d,e,f）を求める
function computeAffineFromTriangles(s0, s1, s2, d0, d1, d2) {
    // Solve for matrix M: [a c e; b d f; 0 0 1] such that M*[sx,sy,1]^T = [dx,dy,1]^T for 3 points
    // Build linear system A * params = B, params = [a,b,c,d,e,f]
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

// 6x6 連立一次方程式を解く（ガウス消去法）
function solve6x6(A, B) {
    // Augmented matrix
    const M = A.map((row, i) => [...row, B[i]]);
    const n = 6;
    for (let i = 0; i < n; i++) {
        // Pivot
        let maxRow = i;
        for (let r = i + 1; r < 2 * n; r++) {
            if (M[r] && Math.abs(M[r][i]) > Math.abs(M[maxRow][i])) maxRow = r;
        }
        if (Math.abs(M[maxRow][i]) < 1e-8) continue;
        [M[i], M[maxRow]] = [M[maxRow], M[i]];
        // Normalize row
        const pivot = M[i][i];
        for (let c = i; c <= n; c++) M[i][c] /= pivot;
        // Eliminate other rows
        for (let r = 0; r < n; r++) {
            if (r === i) continue;
            const factor = M[r][i];
            for (let c = i; c <= n; c++) M[r][c] -= factor * M[i][c];
        }
    }
    return M.map(row => row[n]);
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
    const origWidth = image.width;
    const origHeight = image.height;
    
    // DetResizeForTest相当の処理
    const { resizedImage, ratios, destWidth, destHeight } = resizeImageForDetection(image, 960, 'max');
    
    // 前処理: NormalizeImage相当
    // PaddleOCR(det, DB) は HWC 入力のまま mean/std を適用し、その後 CHW へ変換。
    // Canvas は RGBA なので、BGR順に取り出して NormalizeImage と同じ式で正規化する。
    const normalizedData = normalizeImage(
        resizedImage,
        [0.485, 0.456, 0.406], // mean (B,G,R に対応させて使用)
        [0.229, 0.224, 0.225], // std
        1.0 / 255.0            // scale
    );
    
    // CHW形式に変換
    const chwData = hwc2chw(normalizedData, destHeight, destWidth);
    
    // Tensorを作成 (形状: [1, 3, height, width])
    const tensor = new ort.Tensor('float32', chwData, [1, 3, destHeight, destWidth]);
    
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
    } else if (limitType === 'resize_long') {
        ratio = limitSideLen / Math.max(origHeight, origWidth);
    } else {
        throw new Error('Unsupported limit type');
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
                const hwcIndex = (h * width + w) * 3 + c;
                const chwIndex = c * height * width + h * width + w;
                chwData[chwIndex] = data[hwcIndex];
            }
        }
    }
    
    return chwData;
}

// テキスト検出 (DBPostProcess相当の後処理を含む)
async function detectText(imageTensorInfo) {
    // モデルの入力
    const feeds = {
        'x': imageTensorInfo.tensor
    };
    
    // 推論実行
    const results = await sessionDet.run(feeds);
    
    // 結果を処理 (DBPostProcess相当)
    const output = results['sigmoid_0.tmp_0'] || results[Object.keys(results)[0]];
    const shapeInfo = imageTensorInfo.shapeInfo;

    return postProcessDetection(output, shapeInfo);
}

// 検出結果の後処理 (DBPostProcess相当)
function postProcessDetection(pred, shapeInfo) {
    const [srcH, srcW, ratioH, ratioW] = shapeInfo;
    // ocr_process.md のデフォルト値に合わせる
    const thresh = 0.3;       // det_db_thresh
    const boxThresh = 0.6;    // det_db_box_thresh
    const unclipRatio = 1.5;  // det_db_unclip_ratio
    const maxCandidates = 1000;
    
    // pred形状: [1, 1, H, W]
    const batchSize = pred.dims[0];
    const channels = pred.dims[1];
    const height = pred.dims[2];
    const width = pred.dims[3];
    
    // バッチの最初の要素を処理
    const predData = pred.data; // 長さは H*W (1ch想定)
    const segmentation = new Uint8Array(height * width);
    
    // 二値化
    for (let i = 0; i < height * width; i++) {
        segmentation[i] = predData[i] > thresh ? 1 : 0;
    }
    
    // 連結成分を抽出し、PCAに基づく最小外接矩形 + DBのunclip近似で四角形を生成
    const detectedBoxes = findContoursAndCreateBoxes(
        segmentation, width, height, predData,
        boxThresh, unclipRatio, maxCandidates, srcW, srcH
    );

    return detectedBoxes;
}

// 輪郭検出とボックス作成 (cv2.findContours + DBPostProcess相当の近似)
function findContoursAndCreateBoxes(segmentation, width, height, predData, boxThresh, unclipRatio, maxCandidates, destWidth, destHeight) {
    const boxes = [];
    const visited = new Array(width * height).fill(false);

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const idx = y * width + x;
            if (segmentation[idx] !== 1 || visited[idx]) continue;

            const component = floodFill(segmentation, visited, x, y, width, height);
            if (component.length < 10) continue; // ノイズ除去

            // 平均スコア（fast modeの近似）
            let scoreSum = 0;
            for (const p of component) scoreSum += predData[p.y * width + p.x];
            const score = scoreSum / component.length;
            if (score < boxThresh) continue;

            // PCAで向きを推定し、最小外接矩形（近似）を取得
            const oriented = computeOrientedRect(component);
            if (!oriented) continue;

            // DBのunclip相当: d = area*unclip_ratio/length を矩形に適用
            const unclipped = unclipRectangle(oriented, unclipRatio);
            if (!unclipped) continue;

            // 元画像サイズへスケール
            const scaled = scaleBoxToOriginal(unclipped.points, width, height, destWidth, destHeight);
            if (scaled) boxes.push(scaled);
            if (boxes.length >= maxCandidates) return boxes;
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

// PCA で近似した最小外接矩形（回転矩形）を求める
function computeOrientedRect(component) {
    if (!component || component.length < 4) return null;
    // 質心
    let sx = 0, sy = 0;
    for (const p of component) { sx += p.x; sy += p.y; }
    const n = component.length;
    const cx = sx / n, cy = sy / n;

    // 共分散
    let sxx = 0, syy = 0, sxy = 0;
    for (const p of component) {
        const dx = p.x - cx, dy = p.y - cy;
        sxx += dx * dx; syy += dy * dy; sxy += dx * dy;
    }
    sxx /= n; syy /= n; sxy /= n;

    // 主成分の角度
    const angle = 0.5 * Math.atan2(2 * sxy, sxx - syy);
    const cos = Math.cos(angle), sin = Math.sin(angle);

    // 回転座標系でのAABB
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

    // 矩形4点（回転座標系）
    const halfW = w / 2, halfH = h / 2;
    const cornersLocal = [
        [cxLocal - halfW, cyLocal - halfH], // TL
        [cxLocal + halfW, cyLocal - halfH], // TR
        [cxLocal + halfW, cyLocal + halfH], // BR
        [cxLocal - halfW, cyLocal + halfH], // BL
    ];

    // 元座標系へ戻す
    const corners = cornersLocal.map(([rx, ry]) => [
        cx + cos * rx - sin * ry,
        cy + sin * rx + cos * ry,
    ]);

    return { points: corners, w, h, cx, cy, cos, sin, cxLocal, cyLocal };
}

// DB の unclip を矩形に対して近似適用
function unclipRectangle(rect, unclipRatio) {
    const { w, h, cx, cy, cos, sin, cxLocal, cyLocal } = rect;
    // DB の距離: area*unclip_ratio/length（矩形想定）
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
    
    // 最小サイズチェック
    const width = Math.hypot(scaledPoints[1][0] - scaledPoints[0][0], scaledPoints[1][1] - scaledPoints[0][1]);
    const height = Math.hypot(scaledPoints[3][0] - scaledPoints[0][0], scaledPoints[3][1] - scaledPoints[0][1]);
    
    if (width <= 3 || height <= 3) return null;
    
    return {
        // 参考: predict_det.filter_tag_det_res と同等の4点（時計回り）を期待
        x: Math.min(...scaledPoints.map(p => p[0])),
        y: Math.min(...scaledPoints.map(p => p[1])),
        width: Math.max(...scaledPoints.map(p => p[0])) - Math.min(...scaledPoints.map(p => p[0])),
        height: Math.max(...scaledPoints.map(p => p[1])) - Math.min(...scaledPoints.map(p => p[1])),
        points: orderQuadPoints(scaledPoints)
    };
}

// テキスト認識 (TextRecognizer.__call__相当)
async function recognizeText(croppedImages) {
    const results = [];
    const batchSize = 6; // rec_batch_num
    
    // アスペクト比でソート（処理高速化のため）
    const imageInfos = croppedImages.map((img, index) => ({
        image: img,
        index: index,
        ratio: img.width / img.height
    }));
    
    imageInfos.sort((a, b) => a.ratio - b.ratio);
    
    // optional: 角度分類（180度反転の補正）
    if (sessionCls) {
        for (let i = 0; i < imageInfos.length; i++) {
            const rotated = await maybeRotateByAngleClassifier(imageInfos[i].image);
            if (rotated) imageInfos[i].image = rotated;
        }
    }

    // バッチ処理
    for (let batchStart = 0; batchStart < imageInfos.length; batchStart += batchSize) {
        const batchEnd = Math.min(imageInfos.length, batchStart + batchSize);
        const batch = imageInfos.slice(batchStart, batchEnd);
        
        // バッチ内の最大アスペクト比を計算 (Python版と同じ処理)
        const imgC = 3, imgH = 48, imgW = 320;
        let maxRatio = imgW / imgH; // 初期値
        for (const item of batch) {
            maxRatio = Math.max(maxRatio, item.ratio);
        }
        
        // バッチ内の画像を前処理
        const normalizedBatch = [];
        for (const item of batch) {
            const normalizedImg = resizeNormImgForRecognition(item.image, maxRatio);
            normalizedBatch.push(normalizedImg);
        }
        
        // バッチテンソル作成
        const batchTensor = createBatchTensor(normalizedBatch);
        
        // 推論実行
        const batchResults = await runRecognitionInference(batchTensor);
        
        // 結果をデコード
        const decodedResults = decodeRecognitionResultsCorrect(batchResults);        // 結果を元のインデックス順に戻す
        for (let i = 0; i < batch.length; i++) {
            results[batch[i].index] = decodedResults[i];
        }
    }
    
    return results;
}

// 角度分類を実行し、180度判定なら回転させて返す
async function maybeRotateByAngleClassifier(canvas) {
    try {
        const tensor = preprocessForCls(canvas);
        const feeds = { 'x': tensor };
        const outputs = await sessionCls.run(feeds);
        const out = outputs[Object.keys(outputs)[0]];
        const [b, numClasses] = out.dims; // 1 x 2
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

// 認識用画像前処理 (TextRecognizer.resize_norm_img相当)
function resizeNormImgForRecognition(canvas, maxRatio) {
    const imgC = 3, imgH = 48, imgW = 320; // rec_image_shape
    
    // Python版と同じ処理: imgW = int((imgH * max_wh_ratio))
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
    
    // 実際の最大幅は320
    resizedW = Math.min(resizedW, imgW);
    
    // リサイズ
    const resizeCanvas = document.createElement('canvas');
    const resizeCtx = resizeCanvas.getContext('2d');
    resizeCanvas.width = resizedW;
    resizeCanvas.height = imgH;
    
    resizeCtx.drawImage(canvas, 0, 0, resizedW, imgH);
    
    // 画像データを取得して正規化 (Python版と同じ処理)
    const imageData = resizeCtx.getImageData(0, 0, resizedW, imgH);
    const data = imageData.data;
    
    // Python版のresize_norm_imgの実装に完全に合わせる
    // resized_image = cv2.resize(img, (resized_w, imgH))
    // resized_image = resized_image.astype('float32')
    // resized_image = resized_image.transpose((2, 0, 1)) / 255
    // resized_image -= 0.5; resized_image /= 0.5
    // padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    // padding_im[:, :, 0:resized_w] = resized_image
    
    const paddingIm = new Float32Array(imgC * imgH * imgW); // ゼロで初期化
    
    // resized_imageをCHW形式で作成
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
    const feeds = {
        'x': batchTensor
    };
    
    const results = await sessionRec.run(feeds);
    
    // 出力テンソル取得（モデルによって名前が異なる）
    const outputKey = Object.keys(results)[0];
    return results[outputKey];
}

// 切り抜いた画像をTensorに変換 (認識用) - 削除予定の関数
    const data = output.data;
    if (dims.length !== 3) {
        console.warn('Unexpected recognition output dims:', dims);
        return Array(dims[0] || 0).fill({ text: '', confidence: 0 });
    }

    const B = dims[0]; // batch size
    const T = dims[1]; // sequence length 
    const C = dims[2]; // class number (should equal charDict.length)

    // CTCLabelDecode.__call__相当の処理
    // preds_idx = preds.argmax(axis=2)
    // preds_prob = preds.max(axis=2)
    const predsIdx = new Array(B);
    const predsProb = new Array(B);
    
    for (let b = 0; b < B; b++) {
        predsIdx[b] = new Array(T);
        predsProb[b] = new Array(T);
        
        for (let t = 0; t < T; t++) {
            let maxIdx = 0;
            let maxVal = data[b * T * C + t * C + 0];
            
            for (let c = 1; c < C; c++) {
                const val = data[b * T * C + t * C + c];
                if (val > maxVal) {
                    maxVal = val;
                    maxIdx = c;
                }
            }
            
            predsIdx[b][t] = maxIdx;
            predsProb[b][t] = maxVal;
        }
    }

    
    // CTCLabelDecode.decode相当の処理
    for (let b = 0; b < B; b++) {
        const textIndex = predsIdx[b];
        const textProb = predsProb[b];
        
        // CTCのis_remove_duplicate=True処理
        const selection = new Array(T).fill(true);
        // selection[1:] = text_index[1:] != text_index[:-1] 
        for (let i = 1; i < T; i++) {
            selection[i] = textIndex[i] !== textIndex[i - 1];
        }
        
        // ignored_tokens (blank=0) を除外
        for (let i = 0; i < T; i++) {
            if (textIndex[i] === 0) { // blank token
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
        for (let t = 0; t < T; t++) {
            let maxIndex = 0;
            let maxValue = -Infinity;
            if (layout === 'BTC') {
                // index = b*T*C + t*C + c
                const base = b * T * C + t * C;
                for (let c = 0; c < C; c++) {
                    const v = data[base + c];
                    if (v > maxValue) { maxValue = v; maxIndex = c; }
                }
            } else {
                // BCT: index = b*C*T + c*T + t
                const base = b * C * T + t; // we’ll step by T per class
                for (let c = 0; c < C; c++) {


// 切り抜いた画像をTensorに変換 (認識用) - 削除予定の関数
function croppedImageToTensor(canvas) {
    // 認識モデルの入力サイズ (例: 32x320)
    const recWidth = 320;
    const recHeight = 32;
    
    // Canvasをリサイズ
    const resizedCanvas = document.createElement('canvas');
    const resizedCtx = resizedCanvas.getContext('2d');
    resizedCanvas.width = recWidth;
    resizedCanvas.height = recHeight;
    resizedCtx.drawImage(canvas, 0, 0, recWidth, recHeight);
    
    // 画像データを取得
    const imageData = resizedCtx.getImageData(0, 0, recWidth, recHeight);
    const data = imageData.data;
    
    // RGBに変換 (RGBAからRGBへ) し、正規化
    const rgbData = new Float32Array(recWidth * recHeight * 3);
    for (let i = 0; i < data.length / 4; i++) {
        rgbData[i * 3] = (data[i * 4] / 255.0 - 0.5) / 0.5;       // R
        rgbData[i * 3 + 1] = (data[i * 4 + 1] / 255.0 - 0.5) / 0.5; // G
        rgbData[i * 3 + 2] = (data[i * 4 + 2] / 255.0 - 0.5) / 0.5; // B
    }
    
    // Tensorを作成 (形状: [1, 3, height, width])
    const tensor = new ort.Tensor('float32', rgbData, [1, 3, recHeight, recWidth]);
    return tensor;
}

// 単一テキスト認識 - 削除予定の関数
async function recognizeSingleText(imageTensor) {
    // モデルの入力
    const feeds = {
        'x': imageTensor
    };
    
    // 推論実行
    const results = await sessionRec.run(feeds);
    
    // 結果を処理
    // ここでは簡略化のため、最も確信度の高い文字列を返す
    // 実際のPaddleOCRでは、CTCデコードなどの処理が必要
    const output = results['save_infer_model/scale_0.tmp_1']; // 出力テンソルの名前はモデルによって異なる
    const dataArray = output.data;
    const shape = output.dims;
    
    // 最も確信度の高いインデックスを取得
    let text = '';
    let confidence = 0;
    const seqLength = shape[1];
    const numChars = shape[2];
    
    for (let i = 0; i < seqLength; i++) {
        let maxIndex = 0;
        let maxValue = dataArray[i * numChars];
        
        for (let j = 1; j < numChars; j++) {
            const value = dataArray[i * numChars + j];
            if (value > maxValue) {
                maxValue = value;
                maxIndex = j;
            }
        }
        
        // 確信度を加算 (平均を取る)
        confidence += maxValue;
        
        // 辞書から文字を取得
        if (maxIndex > 0 && maxIndex < charDict.length) {
            text += charDict[maxIndex];
        }
    }
    
    confidence = confidence / seqLength;
    
    return { text, confidence };
}

// 結果の表示
function displayResults(results) {
    // Canvasに結果を描画
    ctx.strokeStyle = 'red';
    ctx.lineWidth = 2;
    ctx.fillStyle = 'red';
    ctx.font = '16px sans-serif';
    
    let resultText = '<h2>OCR Results:</h2><ul>';
    
    for (const result of results) {
        // 矩形を描画
        if (result.box.points && result.box.points.length === 4) {
            // 4点の多角形を描画
            ctx.beginPath();
            ctx.moveTo(result.box.points[0][0], result.box.points[0][1]);
            for (let i = 1; i < result.box.points.length; i++) {
                ctx.lineTo(result.box.points[i][0], result.box.points[i][1]);
            }
            ctx.closePath();
            ctx.stroke();
            
            // テキストを描画
            ctx.fillText(result.text, result.box.points[0][0], result.box.points[0][1] - 10);
        } else {
            // 矩形を描画
            ctx.strokeRect(result.box.x, result.box.y, result.box.width, result.box.height);
            
            // テキストを描画
            ctx.fillText(result.text, result.box.x, result.box.y - 10);
        }
        
        // 結果をHTMLに表示
        resultText += `<li>${result.text} (confidence: ${result.confidence.toFixed(3)})</li>`;
    }
    
    resultText += '</ul>';
    resultDiv.innerHTML = resultText;
}

// 正しいCTCデコード関数 (Python版に忠実)
function decodeRecognitionResultsCorrect(output) {
    const results = [];
    const dims = output.dims;
    const data = output.data;
    
    if (dims.length !== 3) {
        console.warn('Unexpected recognition output dims:', dims);
        return Array(dims[0] || 0).fill({ text: '', confidence: 0 });
    }

    const B = dims[0]; // batch size
    const T = dims[1]; // sequence length 
    const C = dims[2]; // class number (should equal charDict.length)

    // CTCLabelDecode.__call__相当の処理
    // Python版では予測値にsoftmaxが適用されている前提で処理する
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
                logits[c] /= sumExp; // softmax probability
                if (logits[c] > maxProb) {
                    maxProb = logits[c];
                    maxIdx = c;
                }
            }
            
            predsIdx[b][t] = maxIdx;
            predsProb[b][t] = maxProb;
        }
    }
    
    // CTCLabelDecode.decode相当の処理 (is_remove_duplicate=True)
    for (let b = 0; b < B; b++) {
        const textIndex = predsIdx[b];
        const textProb = predsProb[b];
        
        // CTCの重複除去処理: selection[1:] = text_index[1:] != text_index[:-1]
        const selection = new Array(T).fill(true);
        for (let i = 1; i < T; i++) {
            selection[i] = textIndex[i] !== textIndex[i - 1];
        }
        
        // ignored_tokens (blank=0) を除外
        for (let i = 0; i < T; i++) {
            if (textIndex[i] === 0) { // blank token
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
// 2点間距離
function distance(a, b) {
    const dx = a[0] - b[0];
    const dy = a[1] - b[1];
    return Math.hypot(dx, dy);
}

// 4点を TL, TR, BR, BL の順に並べ替える
function orderQuadPoints(points) {
    const pts = points.map(p => ({ x: p[0], y: p[1] }));
    // 左上: x+y が最小, 右下: x+y が最大
    const sumSorted = [...pts].sort((p1, p2) => (p1.x + p1.y) - (p2.x + p2.y));
    const tl = sumSorted[0];
    const br = sumSorted[3];
    // 右上: x-y が最大, 左下: x-y が最小
    const diffSorted = [...pts].sort((p1, p2) => (p1.x - p1.y) - (p2.x - p2.y));
    const bl = diffSorted[0];
    const tr = diffSorted[3];
    return [ [tl.x, tl.y], [tr.x, tr.y], [br.x, br.y], [bl.x, bl.y] ];
}

// 初期化の実行
init();