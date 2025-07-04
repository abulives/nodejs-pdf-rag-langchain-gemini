$(document).ready(function () {
  const loader = $('#loader');
  const uploadResponse = $('#uploadResponse');
  const questionResponse = $('#questionResponse');
  const saveBtn = $('#saveBtn');

  $('#uploadForm').on('submit', function (e) {
    e.preventDefault();
    const formData = new FormData();
    const files = $('#pdfFiles')[0].files;

    $.each(files, function (i, file) {
      formData.append('files', file);
    });

    loader.show();
    uploadResponse.text('');

    $.ajax({
      url: 'http://localhost:3000/api/upload_files',
      method: 'POST',
      data: formData,
      processData: false,
      contentType: false,
      success: function (res) {
        uploadResponse.text(res.message || 'Upload successful!');
      },
      error: function () {
        uploadResponse.text('Error uploading files');
      },
      complete: function () {
        loader.hide();
      }
    });
  });

  $('#questionForm').on('submit', function (e) {
    e.preventDefault();
    const question = $('#questionInput').val();

    loader.show();
    questionResponse.text('');

    $.ajax({
      url: 'http://localhost:3000/api/ask_question',
      method: 'POST',
      contentType: 'application/json',
      data: JSON.stringify({ question }),
      success: function (res) {
        questionResponse.text(res.response || 'Response received.');
        saveBtn.show();
      },
      error: function () {
        questionResponse.text('Error generating response');
      },
      complete: function () {
        loader.hide();
      }
    });
  });
});
