document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const btnPredict = document.getElementById('btn-predict');
    
    const loadingOverlay = document.getElementById('loading-overlay');
    const metricsCard = document.getElementById('metrics-card');
    const gatingCard = document.getElementById('gating-card');
    
    // Result details
    const predClass = document.getElementById('pred-class');
    const predConfidence = document.getElementById('pred-confidence');
    const confMeterFill = document.getElementById('conf-meter-fill');
    const detailedProbs = document.getElementById('detailed-probs');
    
    const vitWeightVal = document.getElementById('vit-weight-val');
    const cnnWeightVal = document.getElementById('cnn-weight-val');
    const gatingBarVit = document.getElementById('gating-bar-vit');
    const gatingBarCnn = document.getElementById('gating-bar-cnn');
    const activeExplainer = document.getElementById('active-explainer');
    
    // Result Images
    const imgInput = document.getElementById('img-input');
    const imgInputBg = document.getElementById('img-input-bg');
    const imgSegmentation = document.getElementById('img-segmentation');
    const imgGradcam = document.getElementById('img-gradcam');
    const imgRollout = document.getElementById('img-rollout');
    
    const maskOpacity = document.getElementById('mask-opacity');
    
    let selectedFile = null;

    // --- Drag and Drop File Handlers ---
    
    // Clicking drop-zone triggers file browser
    dropZone.addEventListener('click', () => fileInput.click());
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.add('drag-over');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.remove('drag-over');
        }, false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });

    function handleFileSelect(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select an image file.');
            return;
        }
        selectedFile = file;
        btnPredict.disabled = false;
        
        // Show local preview immediately
        const reader = new FileReader();
        reader.onload = (e) => {
            imgInput.src = e.target.result;
            imgInputBg.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }

    // --- Opacity Mask Control ---
    maskOpacity.addEventListener('input', (e) => {
        imgSegmentation.style.opacity = e.target.value;
    });

    // --- Model Stage Selector Info updates ---
    const stageSelectEl = document.getElementById('stage-select');
    const stageInfoEl = document.getElementById('stage-info');
    
    const STAGE_DESCRIPTIONS = {
        stage1: {
            title: "Stage 1: TrashNet Classification",
            text: "Optimized for high-accuracy (98.4%) fine-grained classification of clean, single-object waste (cardboard, glass, metal, paper, plastic, trash). No segmentation masks are output."
        },
        stage2: {
            title: "Stage 2: TACO Multi-Task",
            text: "Detects, segments, and classifies all 6 waste types (cardboard, glass, metal, paper, plastic, trash) in diverse real-world cluttered backgrounds (outdoor scenes, parks)."
        },
        stage3: {
            title: "Stage 3: ZeroWaste Industrial",
            text: "Fine-tuned specifically for conveyor belt industrial sorting. Segments metal, plastic, and cardboard. Glass, paper, and general trash are treated as background."
        }
    };
    
    stageSelectEl.addEventListener('change', (e) => {
        const stage = e.target.value;
        const desc = STAGE_DESCRIPTIONS[stage];
        stageInfoEl.innerHTML = `
            <strong>${desc.title}</strong>
            <p>${desc.text}</p>
        `;
    });

    // --- Predict Button Handler ---
    btnPredict.addEventListener('click', () => {
        if (!selectedFile) return;
        
        const stageSelect = document.getElementById('stage-select');
        const formData = new FormData();
        formData.append('image', selectedFile);
        formData.append('synthetic', 'false');
        formData.append('stage', stageSelect.value);
        
        runPrediction(formData);
    });


    // --- API Request Call ---
    function runPrediction(formData) {
        // Show spinner
        loadingOverlay.classList.remove('hidden');
        
        fetch('/api/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => { throw new Error(err.error || 'Server Error'); });
            }
            return response.json();
        })
        .then(data => {
            displayResults(data);
        })
        .catch(error => {
            console.error('Error:', error);
            alert(`Prediction failed: ${error.message}`);
        })
        .finally(() => {
            // Hide spinner
            loadingOverlay.classList.add('hidden');
        });
    }

    // --- Populate Predictions Dashboard ---
    function displayResults(data) {
        // 1. Classification Metrics
        predClass.textContent = data.class;
        predConfidence.textContent = `${(data.confidence * 100).toFixed(1)}%`;
        confMeterFill.style.width = `${data.confidence * 100}%`;
        
        // Detailed Probabilities list
        detailedProbs.innerHTML = '';
        Object.entries(data.detailed_probabilities)
            .sort((a, b) => b[1] - a[1]) // sort descending
            .forEach(([className, prob]) => {
                const row = document.createElement('div');
                row.className = 'prob-row';
                row.innerHTML = `
                    <div class="prob-labels">
                        <span class="class-name">${className}</span>
                        <span>${(prob * 100).toFixed(1)}%</span>
                    </div>
                    <div class="prob-bar-wrapper">
                        <div class="prob-bar-fill" style="width: ${prob * 100}%;"></div>
                    </div>
                `;
                detailedProbs.appendChild(row);
            });
            
        metricsCard.classList.remove('hidden');

        // 2. Gating Weight Info
        const vitWeight = data.gating.alpha_vit;
        const cnnWeight = data.gating.alpha_cnn;
        
        vitWeightVal.textContent = vitWeight.toFixed(2);
        cnnWeightVal.textContent = cnnWeight.toFixed(2);
        
        gatingBarVit.style.width = `${vitWeight * 100}%`;
        gatingBarCnn.style.width = `${cnnWeight * 100}%`;
        
        activeExplainer.textContent = data.gating.method_used;
        gatingCard.classList.remove('hidden');

        // 3. Update Visualizations
        imgInput.src = data.images.input;
        imgInputBg.src = data.images.input;
        imgSegmentation.src = data.images.segmentation;
        imgGradcam.src = data.images.gradcam;
        imgRollout.src = data.images.rollout;
        
        // Match opacity to range input value
        imgSegmentation.style.opacity = maskOpacity.value;
    }
});
