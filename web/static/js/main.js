// main.js

$(document).ready(function () {
    // Initial setup: hide interactive elements
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();
    $('#btn-retry').hide();

    // Delegate click on dynamically generated artist-choice buttons
    $('#result').on('click', '.author-choice', function () {
        const $btn = $(this);
        const guessed = $btn.data('author');
        const actual = $('#actual-author').data('author');

        // Disable all choice buttons after a guess
        $('.author-choice').prop('disabled', true);

        // Highlight the clicked button: green if correct, red if wrong
        if (guessed === actual) {
            $btn.removeClass('btn-outline-primary')
                .addClass('btn-success');
            $('#guess-feedback').html('<span class="text-success">Correct!</span>');
        } else {
            $btn.removeClass('btn-outline-primary')
                .addClass('btn-danger');
            const actualText = actual.replace(/_/g, ' ');
            $('#guess-feedback').html(
                `<span class="text-danger">Nope—it was ${actualText}.</span>`
            );
        }
    });

    // Function to preview the uploaded image
    function readURL(input) {
        if (input.files && input.files[0]) {
            const reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview')
                    .attr('src', e.target.result)   // Set <img> src to the uploaded file
                    .hide()
                    .fadeIn(650);
            };
            reader.readAsDataURL(input.files[0]);
        }
    }

    // When a file is selected, show preview and the "Stylize!" button
    $('#imageUpload').change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').hide().empty();
        $('#btn-retry').hide();
        readURL(this);
    });

    // Handle click on "Stylize!" button
    $('#btn-predict').click(function () {
        const form_data = new FormData($('#upload-file')[0]);

        // Hide the predict button and show the loader spinner
        $(this).hide();
        $('.loader').show();

        // Send the image to the server for style transfer
        $.ajax({
            type: 'POST',
            url: '/',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            success: function (data) {
                // Hide loader, insert result HTML, then show retry button
                $('.loader').hide();
                $('#result').html(data).fadeIn(600);
                $('#btn-retry').show();
            },
            error: function (xhr) {
                // On error, hide loader and display error message
                $('.loader').hide();
                $('#result')
                    .html(`<p class="text-danger">Error: ${xhr.responseText}</p>`)
                    .fadeIn(600);
            }
        });
    });

    // Handle click on "Try Another Image" button
    $('#btn-retry').click(function () {
        // Reset upload form
        $('#upload-file')[0].reset();
        // Clear the preview image
        $('#imagePreview').attr('src', '').hide();
        // Hide all interactive sections and buttons
        $('.image-section').hide();
        $('#btn-predict').hide();
        $('#result').hide().empty();
        // Hide the retry button itself
        $(this).hide();
    });
});
