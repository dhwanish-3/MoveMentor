// const startButton = document.getElementById('startBtn');
// const stopButton = document.getElementById('stopBtn');
const recordBtn = document.getElementById('recordBtn');
let mediaRecorder;
let recordedChunks = [];

const videoPlayer = document.getElementById("videoPlayer");
const myVideo = document.getElementById("myVideo");

const getResultsBtn = document.querySelector(".get-results");

// stopButton.disabled = true;
const videoElement = document.getElementById('videoElement');

navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => {
    videoElement.srcObject = stream;
  })
  .catch(error => {
    console.error('Error accessing media devices.', error);
  });
  

recordBtn.addEventListener('click', async () => {
    console.log("inside start recording");
    // startButton.disabled = true;
    // startButton.textContent = 'Recording...';
    if (recordBtn.textContent === 'Start Recording') {
        recordedChunks = [];
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        mediaRecorder = new MediaRecorder(stream);

        recordBtn.textContent = 'Stop Recording';
        console.log('MediaRecorder started', mediaRecorder);

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
    } else {
        mediaRecorder.stop();
        recordBtn.textContent = 'Start Recording';
        console.log('Recorded Blobs: ', recordedChunks);
      }
    // startButton.disabled = true;
    // stopButton.disabled = false;
});

// stopButton.addEventListener('click', () => {
//     mediaRecorder.stop();
//     startButton.disabled = true;
//     stopButton.disabled = true;
//     getResultsBtn.innerHTML = `<button class="button" id="getResults">Get Results</button>`;
// });

function stopVideo() {
    videoPlayer.style.display = "none";
    myVideo.src = null;
}

function  playVideo(file){
    myVideo.src = file; 
    videoPlayer.style.display = "block";
}