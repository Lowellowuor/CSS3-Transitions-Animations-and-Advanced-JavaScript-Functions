// DOM Elements
const trainBtn = document.getElementById('train-btn');
const pauseBtn = document.getElementById('pause-btn');
const resetBtn = document.getElementById('reset-btn');
const modelType = document.getElementById('model-type');
const dataset = document.getElementById('dataset');
const learningRate = document.getElementById('learning-rate');
const lrValue = document.getElementById('lr-value');
const batchSize = document.getElementById('batch-size');
const darkMode = document.getElementById('dark-mode');
const trainingProgress = document.getElementById('training-progress');
const statusMessage = document.getElementById('status-message');
const neuralNetwork = document.getElementById('neural-network');
const lossLine = document.getElementById('loss-line');
const tabs = document.querySelectorAll('.tab');
const tabContents = document.querySelectorAll('.tab-content');
const featureMaps = document.getElementById('feature-maps');
const attentionVis = document.getElementById('attention-vis');
const confusionMatrix = document.getElementById('confusion-matrix');
const trainLoss = document.getElementById('train-loss');
const valLoss = document.getElementById('val-loss');
const accuracy = document.getElementById('accuracy');
const epochDisplay = document.getElementById('epoch');

// State variables
let isTraining = false;
let trainingInterval;
let currentEpoch = 0;
let maxEpochs = 10;
let lossHistory = [];
let valLossHistory = [];
let accuracyHistory = [];
let userPreferences = {};

// Initialize the application
function init() {
    loadPreferences();
    setupEventListeners();
    renderNeuralNetwork();
    setupConfusionMatrix();
    generateFeatureMaps();
    setupAttentionVisualization();
}

// Load user preferences from localStorage
function loadPreferences() {
    const savedPrefs = localStorage.getItem('dlVisualizerPrefs');
    if (savedPrefs) {
        userPreferences = JSON.parse(savedPrefs);
        
        // Apply preferences
        if (userPreferences.modelType) modelType.value = userPreferences.modelType;
        if (userPreferences.dataset) dataset.value = userPreferences.dataset;
        if (userPreferences.learningRate) {
            learningRate.value = userPreferences.learningRate;
            lrValue.textContent = userPreferences.learningRate;
        }
        if (userPreferences.batchSize) batchSize.value = userPreferences.batchSize;
        if (userPreferences.darkMode) {
            darkMode.checked = userPreferences.darkMode;
            toggleDarkMode();
        }
    }
}

// Save user preferences to localStorage
function savePreferences() {
    userPreferences = {
        modelType: modelType.value,
        dataset: dataset.value,
        learningRate: learningRate.value,
        batchSize: batchSize.value,
        darkMode: darkMode.checked
    };
    
    localStorage.setItem('dlVisualizerPrefs', JSON.stringify(userPreferences));
}

// Set up event listeners
function setupEventListeners() {
    trainBtn.addEventListener('click', startTraining);
    pauseBtn.addEventListener('click', togglePause);
    resetBtn.addEventListener('click', resetTraining);
    
    learningRate.addEventListener('input', function() {
        lrValue.textContent = this.value;
        savePreferences();
    });
    
    modelType.addEventListener('change', function() {
        renderNeuralNetwork();
        savePreferences();
    });
    
    dataset.addEventListener('change', savePreferences);
    batchSize.addEventListener('change', savePreferences);
    
    darkMode.addEventListener('change', function() {
        toggleDarkMode();
        savePreferences();
    });
    
    // Tab switching
    tabs.forEach(tab => {
        tab.addEventListener('click', function() {
            const tabId = this.getAttribute('data-tab');
            
            // Update active tab
            tabs.forEach(t => t.classList.remove('active'));
            this.classList.add('active');
            
            // Show corresponding content
            tabContents.forEach(content => {
                content.classList.remove('active');
                if (content.id === `${tabId}-tab`) {
                    content.classList.add('active');
                }
            });
        });
    });
}

// Toggle dark/light mode
function toggleDarkMode() {
    if (darkMode.checked) {
        document.documentElement.style.setProperty('--bg-color', '#121212');
        document.documentElement.style.setProperty('--text-color', '#e1e1e1');
    } else {
        document.documentElement.style.setProperty('--bg-color', '#f5f5f5');
        document.documentElement.style.setProperty('--text-color', '#333333');
    }
}

// Render neural network visualization based on selected model type
function renderNeuralNetwork() {
    neuralNetwork.innerHTML = '';
    
    let layers = [];
    let connections = [];
    
    switch (modelType.value) {
        case 'cnn':
            layers = [
                ['Input', 3],  // RGB channels
                ['Conv1', 8],
                ['Pool1', 8],
                ['Conv2', 16],
                ['Pool2', 16],
                ['Flatten', 1],
                ['Dense', 64],
                ['Output', 10]
            ];
            break;
            
        case 'rnn':
            layers = [
                ['Input', 1],
                ['Embed', 8],
                ['LSTM', 16],
                ['Dense', 32],
                ['Output', 10]
            ];
            break;
            
        case 'transformer':
            layers = [
                ['Input', 1],
                ['Embed', 8],
                ['Attention', 8],
                ['FFN', 8],
                ['Norm', 8],
                ['Output', 10]
            ];
            break;
            
        case 'mlp':
        default:
            layers = [
                ['Input', 4],
                ['Hidden1', 8],
                ['Hidden2', 6],
                ['Output', 3]
            ];
    }
    
    // Create layers
    layers.forEach((layer, layerIdx) => {
        const layerDiv = document.createElement('div');
        layerDiv.className = 'layer';
        layerDiv.setAttribute('data-layer', layer[0]);
        
        // Create neurons
        for (let i = 0; i < layer[1]; i++) {
            const neuron = document.createElement('div');
            neuron.className = 'neuron';
            neuron.textContent = layerIdx === 0 ? 'I' : layerIdx === layers.length - 1 ? 'O' : '';
            neuron.setAttribute('data-neuron', `${layer[0]}-${i}`);
            layerDiv.appendChild(neuron);
            
            // Store positions for connections
            if (layerIdx > 0) {
                const prevLayer = layers[layerIdx - 1];
                for (let j = 0; j < prevLayer[1]; j++) {
                    connections.push({
                        from: `${prevLayer[0]}-${j}`,
                        to: `${layer[0]}-${i}`
                    });
                }
            }
        }
        
        neuralNetwork.appendChild(layerDiv);
    });
    
    // Create connections (but don't show them yet)
    connections.forEach(conn => {
        const connection = document.createElement('div');
        connection.className = 'connection';
        connection.setAttribute('data-connection', `${conn.from}-${conn.to}`);
        neuralNetwork.appendChild(connection);
    });
    
    // Position connections after a short delay to allow layout to settle
    setTimeout(positionConnections, 100);
}

// Position connections between neurons
function positionConnections() {
    document.querySelectorAll('.connection').forEach(conn => {
        const connId = conn.getAttribute('data-connection');
        const [fromId, toId] = connId.split('-').slice(-2);
        
        const fromNeuron = document.querySelector(`[data-neuron="${fromId}"]`);
        const toNeuron = document.querySelector(`[data-neuron="${toId}"]`);
        
        if (fromNeuron && toNeuron) {
            const fromRect = fromNeuron.getBoundingClientRect();
            const toRect = toNeuron.getBoundingClientRect();
            const networkRect = neuralNetwork.getBoundingClientRect();
            
            const fromX = fromRect.left + fromRect.width/2 - networkRect.left;
            const fromY = fromRect.top + fromRect.height/2 - networkRect.top;
            const toX = toRect.left + toRect.width/2 - networkRect.left;
            const toY = toRect.top + toRect.height/2 - networkRect.top;
            
            const length = Math.sqrt(Math.pow(toX - fromX, 2) + Math.pow(toY - fromY, 2));
            const angle = Math.atan2(toY - fromY, toX - fromX);
            
            conn.style.width = `${length}px`;
            conn.style.left = `${fromX}px`;
            conn.style.top = `${fromY}px`;
            conn.style.transform = `rotate(${angle}rad)`;
        }
    });
}

// Start training simulation
function startTraining() {
    if (isTraining) return;
    
    isTraining = true;
    trainBtn.disabled = true;
    pauseBtn.disabled = false;
    currentEpoch = 0;
    lossHistory = [];
    valLossHistory = [];
    accuracyHistory = [];
    
    // Clear previous training visualization
    document.querySelectorAll('.chart-point').forEach(el => el.remove());
    document.querySelectorAll('.data-flow').forEach(el => el.remove());
    
    // Reset progress
    trainingProgress.style.width = '0%';
    statusMessage.textContent = 'Initializing training...';
    
    // Animate connections appearing
    animateConnections();
    
    // Start training loop
    trainingInterval = setInterval(updateTraining, 1500);
}

// Toggle pause/resume training
function togglePause() {
    if (isTraining) {
        clearInterval(trainingInterval);
        isTraining = false;
        pauseBtn.textContent = 'Resume';
        statusMessage.textContent = `Training paused at epoch ${currentEpoch}`;
    } else {
        isTraining = true;
        pauseBtn.textContent = 'Pause';
        statusMessage.textContent = `Resuming training...`;
        trainingInterval = setInterval(updateTraining, 1500);
    }
}

// Reset training
function resetTraining() {
    clearInterval(trainingInterval);
    isTraining = false;
    currentEpoch = 0;
    lossHistory = [];
    valLossHistory = [];
    accuracyHistory = [];
    
    trainBtn.disabled = false;
    pauseBtn.disabled = true;
    pauseBtn.textContent = 'Pause';
    
    trainingProgress.style.width = '0%';
    statusMessage.textContent = 'Ready to train model...';
    
    // Clear visualizations
    document.querySelectorAll('.chart-point').forEach(el => el.remove());
    document.querySelectorAll('.data-flow').forEach(el => el.remove());
    lossLine.style.transform = 'scaleX(0)';
    
    // Reset metrics
    trainLoss.textContent = '0.000';
    valLoss.textContent = '0.000';
    accuracy.textContent = '0.00%';
    epochDisplay.textContent = '0';
    
    // Hide all connections
    document.querySelectorAll('.connection').forEach(conn => {
        conn.style.opacity = '0';
    });
    
    // Reset feature maps
    document.querySelectorAll('.feature-map').forEach(map => {
        map.classList.remove('visible');
    });
    
    // Reset attention visualization
    document.querySelectorAll('.attention-head, .attention-line').forEach(el => {
        el.style.transform = 'scale(0)';
    });
    
    // Reset confusion matrix
    document.querySelectorAll('.matrix-cell').forEach((cell, i) => {
        if (i >= 4) {
            cell.textContent = '0';
            cell.style.backgroundColor = '';
        }
        cell.classList.remove('visible');
    });
}

// Update training progress
function updateTraining() {
    if (currentEpoch >= maxEpochs) {
        finishTraining();
        return;
    }
    
    currentEpoch++;
    epochDisplay.textContent = currentEpoch;
    
    // Simulate training metrics
    const progress = currentEpoch / maxEpochs;
    trainingProgress.style.width = `${progress * 100}%`;
    
    // Generate loss values that generally decrease but with some noise
    const baseLoss = Math.max(0.1, 2.0 * Math.pow(0.5, currentEpoch/2));
    const noise = 0.1 * Math.random();
    const loss = baseLoss + noise;
    lossHistory.push(loss);
    
    // Validation loss is similar but with less improvement
    const valBaseLoss = Math.max(0.15, 2.2 * Math.pow(0.6, currentEpoch/2));
    const valNoise = 0.15 * Math.random();
    const valLossValue = valBaseLoss + valNoise;
    valLossHistory.push(valLossValue);
    
    // Accuracy improves over time
    const acc = Math.min(98, 10 + 88 * (1 - Math.pow(0.7, currentEpoch/2)) + 2 * Math.random());
    accuracyHistory.push(acc);
    
    // Update displays
    trainLoss.textContent = loss.toFixed(3);
    valLoss.textContent = valLossValue.toFixed(3);
    accuracy.textContent = acc.toFixed(2) + '%';
    
    // Update loss chart
    updateLossChart();
    
    // Show data flowing through network
    simulateDataFlow();
    
    // Update status message
    statusMessage.textContent = `Training epoch ${currentEpoch}/${maxEpochs} - Loss: ${loss.toFixed(4)}`;
    
    // Update feature maps as training progresses
    if (currentEpoch % 2 === 0) {
        updateFeatureMaps();
    }
    
    // Update attention visualization for transformer
    if (modelType.value === 'transformer' && currentEpoch % 3 === 0) {
        updateAttentionVisualization();
    }
    
    // Update confusion matrix in later epochs
    if (currentEpoch > maxEpochs / 2) {
        updateConfusionMatrix();
    }
}

// Finish training
function finishTraining() {
    clearInterval(trainingInterval);
    isTraining = false;
    trainBtn.disabled = false;
    pauseBtn.disabled = true;
    
    statusMessage.textContent = `Training completed! Final loss: ${lossHistory[lossHistory.length-1].toFixed(4)}`;
    
    // Pulse the output neurons
    document.querySelectorAll('.layer:last-child .neuron').forEach(neuron => {
        neuron.classList.add('neuron-pulse');
    });
}

// Animate connections appearing
function animateConnections() {
    const connections = document.querySelectorAll('.connection');
    connections.forEach((conn, i) => {
        setTimeout(() => {
            conn.style.opacity = '0.3';
        }, i * 20);
    });
}

// Update loss chart
function updateLossChart() {
    const chartHeight = 300;
    const chartWidth = document.querySelector('.loss-chart').offsetWidth;
    const pointSpacing = chartWidth / maxEpochs;
    
    // Clear previous points (except the first few to show progression)
    if (currentEpoch > 5) {
        document.querySelectorAll('.chart-point').forEach((el, i) => {
            if (i < currentEpoch - 5) el.remove();
        });
    }
    
    // Add new point
    const point = document.createElement('div');
    point.className = 'chart-point';
    point.style.left = `${currentEpoch * pointSpacing}px`;
    
    // Scale loss to fit chart (assuming max loss of 2.5)
    const loss = lossHistory[currentEpoch - 1];
    const scaledLoss = Math.min(1, loss / 2.5);
    point.style.bottom = `${scaledLoss * chartHeight}px`;
    
    document.querySelector('.loss-chart').appendChild(point);
    
    // Animate point appearing
    setTimeout(() => {
        point.style.transform = 'translate(-50%, 50%) scale(1)';
    }, 10);
    
    // Update loss line
    lossLine.style.transform = `scaleX(${currentEpoch / maxEpochs})`;
}

// Simulate data flowing through the network
function simulateDataFlow() {
    const inputNeurons = document.querySelectorAll('.layer:first-child .neuron');
    const outputNeurons = document.querySelectorAll('.layer:last-child .neuron');
    
    // Create data points at random input neurons
    inputNeurons.forEach(neuron => {
        if (Math.random() > 0.7) {
            const dataPoint = document.createElement('div');
            dataPoint.className = 'data-flow';
            
            const neuronRect = neuron.getBoundingClientRect();
            const networkRect = neuralNetwork.getBoundingClientRect();
            
            dataPoint.style.left = `${neuronRect.left + neuronRect.width/2 - networkRect.left}px`;
            dataPoint.style.top = `${neuronRect.top + neuronRect.height/2 - networkRect.top}px`;
            
            neuralNetwork.appendChild(dataPoint);
            
            // Choose a random output neuron as target
            const targetNeuron = outputNeurons[Math.floor(Math.random() * outputNeurons.length)];
            const targetRect = targetNeuron.getBoundingClientRect();
            
            const tx = targetRect.left + targetRect.width/2 - neuronRect.left - neuronRect.width/2;
            const ty = targetRect.top + targetRect.height/2 - neuronRect.top - neuronRect.height/2;
            
            dataPoint.style.setProperty('--tx', `${tx}px`);
                        dataPoint.style.setProperty('--ty', `${ty}px`);
                        dataPoint.style.transform = `translate(var(--tx), var(--ty))`;
                        
                        // Remove data point after animation
                        setTimeout(() => dataPoint.remove(), 1000);
                    }
                });
            }