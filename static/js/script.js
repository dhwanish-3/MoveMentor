const videoInput = document.getElementById('video-select');
const uploadForm = document.getElementById('upload-form');
const uploading = document.querySelector('.upload');

videoInput.addEventListener('change', () => {
    const selectedFile = videoInput.files[0];
    if (selectedFile) {
      uploading.textContent = 'Uploading...';
      uploadForm.submit();
    }
});
