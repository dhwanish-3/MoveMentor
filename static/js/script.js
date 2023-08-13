const videoInput = document.getElementById('video-select');
const uploadForm = document.getElementById('upload-form');
const uploading = document.querySelector('.upload');

videoInput.addEventListener('change', () => {
    const selectedFile = videoInput.files[0];
    console.log("inside");
    if (selectedFile) {
        console.log("inside if");
      uploading.textContent = 'Uploading...';
      uploadForm.submit();
    }
    console.log("outside if");
});

const startButton = document.getElementById('startRecording');
const stopButton = document.getElementById('stopRecording');
let mediaRecorder;
let recordedChunks = [];

startButton.addEventListener('click', async () => {
    recordedChunks = [];
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
    mediaRecorder = new MediaRecorder(stream);
    
    mediaRecorder.ondataavailable = event => {
        if (event.data.size > 0) {
            recordedChunks.push(event.data);
        }
    };
    
    mediaRecorder.onstop = () => {
        const blob = new Blob(recordedChunks, { type: 'video/webm' });
        const formData = new FormData();
        formData.append('video', blob);
        
        // Send the recorded video to the server using AJAX
        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/save_video', true);
        xhr.onreadystatechange = () => {
            if (xhr.readyState === 4 && xhr.status === 200) {
                console.log('Video saved on server');
            }
        };
        xhr.send(formData);
    };
    
    mediaRecorder.start();
    startButton.disabled = true;
    stopButton.disabled = false;
});

stopButton.addEventListener('click', () => {
    mediaRecorder.stop();
    startButton.disabled = false;
    stopButton.disabled = true;
});

const videoPlayer = document.getElementById("videoPlayer");
const myVideo = document.getElementById("myVideo");

function stopVideo() {
    videoPlayer.style.display = "none";
    myVideo.src = null;
}

function  playVideo(file){
    myVideo.src = file; 
    videoPlayer.style.display = "block";
}