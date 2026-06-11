const questionInput = document.getElementById('question');
const queryBtn = document.getElementById('queryBtn');
const loading = document.getElementById('loading');
const results = document.getElementById('results');
const error = document.getElementById('error');

// Upload elements
const uploadBox = document.getElementById('uploadBox');
const fileInput = document.getElementById('fileInput');
const uploadProgress = document.getElementById('uploadProgress');
const uploadStatus = document.getElementById('uploadStatus');
const uploadedFiles = document.getElementById('uploadedFiles');

// Tab elements
const tabBtns = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');

const API_BASE = window.location.origin;

// Event Listeners
queryBtn.addEventListener('click', performQuery);
questionInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') performQuery();
});

// Tab switching
tabBtns.forEach(btn => {
    btn.addEventListener('click', (e) => {
        const tabName = e.target.dataset.tab;
        switchTab(tabName);
    });
});

// File upload
uploadBox.addEventListener('click', () => fileInput.click());
uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.classList.add('drag-over');
});

uploadBox.addEventListener('dragleave', () => {
    uploadBox.classList.remove('drag-over');
});

uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.classList.remove('drag-over');
    handleFiles(e.dataTransfer.files);
});

fileInput.addEventListener('change', (e) => {
    handleFiles(e.target.files);
});

// Functions
function switchTab(tabName) {
    // Update buttons
    tabBtns.forEach(btn => {
        btn.classList.remove('active');
        if (btn.dataset.tab === tabName) btn.classList.add('active');
    });

    // Update content
    tabContents.forEach(content => {
        content.classList.remove('active');
        if (content.id === `${tabName}-tab`) content.classList.add('active');
    });
}

async function performQuery() {
    const question = questionInput.value.trim();
    
    if (!question) {
        showError('Please enter a question');
        return;
    }

    clearMessages();
    showLoading(true);

    try {
        const response = await fetch(`${API_BASE}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                question: question,
                filters: null
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        displayResults(data);
    } catch (err) {
        showError(`Error: ${err.message}`);
    } finally {
        showLoading(false);
    }
}

function displayResults(data) {
    // Display answer
    document.getElementById('answer').textContent = data.answer || 'No answer generated.';

    // Display sources
    const sourcesContainer = document.getElementById('sources');
    sourcesContainer.innerHTML = '';
    if (data.sources && data.sources.length > 0) {
        data.sources.forEach((source, index) => {
            const sourceItem = document.createElement('div');
            sourceItem.className = 'source-item';
            sourceItem.innerHTML = `
                <strong>Source ${index + 1} (Score: ${source.score ? source.score.toFixed(3) : 'N/A'})</strong>
                <p>${source.text ? source.text.substring(0, 200) + '...' : 'No text available'}</p>
            `;
            sourcesContainer.appendChild(sourceItem);
        });
    } else {
        sourcesContainer.innerHTML = '<p style="color: #6b7280;">No sources found.</p>';
    }

    // Display intent
    const intentContainer = document.getElementById('intent');
    if (data.intent) {
        intentContainer.textContent = JSON.stringify(data.intent, null, 2);
    }

    // Display token usage
    const tokensContainer = document.getElementById('tokens');
    tokensContainer.innerHTML = '';
    if (data.token_usage) {
        ['input_tokens', 'output_tokens', 'total_tokens'].forEach(key => {
            const tokenItem = document.createElement('div');
            tokenItem.className = 'token-item';
            tokenItem.innerHTML = `
                <strong>${key.replace(/_/g, ' ')}</strong>
                <span>${data.token_usage[key]}</span>
            `;
            tokensContainer.appendChild(tokenItem);
        });
    }

    results.classList.remove('hidden');
}

async function handleFiles(files) {
    if (!files || files.length === 0) return;

    for (const file of files) {
        // Validate file type
        const validTypes = ['.pdf', '.json', '.csv'];
        const fileExt = '.' + file.name.split('.').pop().toLowerCase();
        
        if (!validTypes.includes(fileExt)) {
            showUploadError(`Invalid file type: ${file.name}. Supported: PDF, JSON, CSV`);
            continue;
        }

        // Upload file
        await uploadFile(file);
    }
}

async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    try {
        showUploadProgress(file.name);

        const xhr = new XMLHttpRequest();

        xhr.upload.addEventListener('progress', (e) => {
            if (e.lengthComputable) {
                const percentComplete = (e.loaded / e.total) * 100;
                updateProgressBar(percentComplete);
            }
        });

        await new Promise((resolve, reject) => {
            xhr.addEventListener('load', () => {
                if (xhr.status === 200) {
                    const response = JSON.parse(xhr.responseText);
                    showUploadSuccess(`Successfully uploaded ${file.name}`);
                    addUploadedFile(file.name, file.size, 'success');
                    resolve();
                } else {
                    const response = JSON.parse(xhr.responseText);
                    throw new Error(response.detail || 'Upload failed');
                }
            });

            xhr.addEventListener('error', reject);

            xhr.open('POST', `${API_BASE}/upload`);
            xhr.send(formData);
        });
    } catch (err) {
        showUploadError(`Error uploading ${file.name}: ${err.message}`);
        addUploadedFile(file.name, file.size, 'error');
    }
}

function showUploadProgress(fileName) {
    document.getElementById('progressFileName').textContent = fileName;
    document.getElementById('progressPercent').textContent = '0%';
    document.getElementById('progressFill').style.width = '0%';
    uploadProgress.classList.remove('hidden');
    uploadStatus.classList.add('hidden');
}

function updateProgressBar(percent) {
    document.getElementById('progressFill').style.width = percent + '%';
    document.getElementById('progressPercent').textContent = Math.round(percent) + '%';
}

function showUploadSuccess(message) {
    uploadProgress.classList.add('hidden');
    uploadStatus.classList.remove('hidden');
    uploadStatus.classList.remove('error');
    uploadStatus.classList.add('success');
    document.getElementById('statusMessage').textContent = '✅ ' + message;
}

function showUploadError(message) {
    uploadProgress.classList.add('hidden');
    uploadStatus.classList.remove('hidden');
    uploadStatus.classList.remove('success');
    uploadStatus.classList.add('error');
    document.getElementById('statusMessage').textContent = '❌ ' + message;
}

function addUploadedFile(fileName, fileSize, status) {
    const fileItem = document.createElement('div');
    fileItem.className = 'file-item';
    
    const ext = fileName.split('.').pop().toUpperCase();
    const iconMap = { PDF: '📄', JSON: '📊', CSV: '📈' };
    const icon = iconMap[ext] || '📁';

    fileItem.innerHTML = `
        <div class="file-icon">${icon}</div>
        <div class="file-name">${fileName}</div>
        <div class="file-size">${formatFileSize(fileSize)}</div>
        <div class="file-status ${status}">${status === 'success' ? '✓ Indexed' : '✗ Error'}</div>
    `;

    uploadedFiles.appendChild(fileItem);
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

function showError(message) {
    error.textContent = message;
    error.classList.remove('hidden');
}

function showLoading(isLoading) {
    if (isLoading) {
        loading.classList.remove('hidden');
    } else {
        loading.classList.add('hidden');
    }
}

function clearMessages() {
    error.classList.add('hidden');
    results.classList.add('hidden');
}

// Focus on input field on page load
window.addEventListener('load', () => {
    questionInput.focus();
});
