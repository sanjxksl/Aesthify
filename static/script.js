function readURL(input, previewContainer, previewImage, showButton) {
    if (input.files && input.files[0]) {
        const reader = new FileReader();
        reader.onload = function(e) {
            $(previewImage).attr('src', e.target.result);
            $(previewContainer).show();
            $(showButton).show();
        };
        reader.readAsDataURL(input.files[0]);
    }
}

function resizeImage(file, callback) {
    const maxWidth = 800;
    const maxHeight = 800;

    const reader = new FileReader();
    reader.onload = function(event) {
        const img = new Image();
        img.onload = function() {
            let canvas = document.createElement('canvas');
            let ctx = canvas.getContext('2d');

            let width = img.width;
            let height = img.height;

            if (width > maxWidth || height > maxHeight) {
                const scale = Math.min(maxWidth / width, maxHeight / height);
                width *= scale;
                height *= scale;
            }

            canvas.width = width;
            canvas.height = height;
            ctx.drawImage(img, 0, 0, width, height);

            const base64Compressed = canvas.toDataURL('image/jpeg', 0.85);  // smaller file
            callback(base64Compressed);
        };
        img.src = event.target.result;
    };
    reader.readAsDataURL(file);
}

$(document).ready(function () {
    // Toggle for Upload Option (Evaluation)
    $('#uploadOption').click(function() {
        $('#uploadSection').show();
        $('#captureSection').hide();
        $('#imagePreviewContainer').hide();
        $('#evaluateButton').hide();
        $('#loadingText').hide();
    });
    
    // Toggle for Capture Option (Evaluation)
    $('#captureOption').click(function() {
        $('#uploadSection').hide();
        $('#imagePreviewContainer').hide();
        $('#evaluateButton').hide();
        $('#captureSection').show();
        $('#startCameraButton').show();
        $('#captureButton').hide();
        $('#video').hide();
        $('#loadingText').hide();

        $('#canvas').hide();
    });
    
    // Upload: open file dialog for evaluation
    $('#uploadButton').click(function() {
        $('#imageInput').click();
    });
    
    // Handle file upload and preview (Evaluation)
    $('#imageInput').change(function(event) {
        const file = event.target.files[0];
        if (file) {
            resizeImage(file, function(base64Compressed) {
                $('#previewImage').attr('src', base64Compressed);
                $('#imagePreviewContainer').show();
                $('#evaluateButton').show();
            });
        }
    });    
    
    // Start camera for evaluation
    $('#startCameraButton').click(function() {
        navigator.mediaDevices.getUserMedia({ video: true })
          .then(function(stream) {
            window.stream = stream;
            $('#video').show();
            $('#captureButton').show();
            $('#startCameraButton').hide();
            $('#video')[0].srcObject = stream;
          })
          .catch(function(err) {
            console.error("Error accessing camera: " + err);
          });
    });
    
    // Capture image from camera (Evaluation)
    $('#captureButton').click(function() {
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const context = canvas.getContext('2d');
        canvas.width = 512;
        canvas.height = 512;
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        if (window.stream) {
          window.stream.getTracks().forEach(track => track.stop());
        }
        $('#video').hide();
        $('#canvas').show();
        $('#evaluateButton').show();
    });
    
    // Evaluate Button: AJAX call to /evaluate; build table for scores.
    $('#evaluateButton').click(function() {
        $('#loadingText').show();
        const imgData = $('#previewImage').attr('src');
        if (!imgData) {
            alert("Please select or capture an image first.");
            return;
        }
        $.ajax({
            type: "POST",
            url: "/evaluate",
            data: { image_data: imgData },
            success: function(response) {
                if (response.error) {
                    console.error("Evaluation error:", response.error);
                    alert("Error: " + response.error);
                    return;
                }
                // Build a table for aesthetic scores:
                let evalTable = `
                  <table class="table table-bordered" style="background: url('/static/images/tablebg.png') no-repeat center center; background-size: cover;">
                    <thead>
                      <tr>
                        <th>Score Type</th>
                        <th>Value</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr><td>Balance Score</td><td>${response.balance_score.toFixed(2)}</td></tr>
                      <tr><td>Proportion Score</td><td>${response.proportion_score.toFixed(2)}</td></tr>
                      <tr><td>Symmetry Score</td><td>${response.symmetry_score.toFixed(2)}</td></tr>
                      <tr><td>Simplicity Score</td><td>${response.simplicity_score.toFixed(2)}</td></tr>
                      <tr><td>Harmony Score</td><td>${response.harmony_score.toFixed(2)}</td></tr>
                      <tr><td>Contrast Score</td><td>${response.contrast_score.toFixed(2)}</td></tr>
                      <tr><td>Unity Score</td><td>${response.unity_score.toFixed(2)}</td></tr>
                      <tr><td>Final Aesthetic Score</td><td>${response.average_aesthetic_value.toFixed(2)}</td></tr>
                    </tbody>
                  </table>
                `;
                $('#evalResultText').html(evalTable);
                if (response.labeled_image) {
                    $('#labeledImageResult').attr('src', response.labeled_image).show();
                }   
                $('#evaluationResult').show();
                $('#labeledImageContainer').show();
                $('#loadingText').hide();             
            },
            error: function(xhr, status, error) {
                console.error("Evaluation request error:", error);
                alert("An error occurred while processing the evaluation request. Please try again.");
            }
        });
    });
});

