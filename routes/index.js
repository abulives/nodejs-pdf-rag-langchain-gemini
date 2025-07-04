var express = require('express');
const fs = require('fs');
var router = express.Router();
const multer = require('multer');
const upload = multer({ dest: 'uploads/' });
const geminiService = require('../services/geminiService');

/* GET home page. */
router.get('/', function (req, res, next) {
  res.render('index', { title: 'Gen AI API' });
});

router.post('/upload_files', upload.array('files'), async function (req, res, next) {
  try {
    await geminiService.uploadPDFs(req.files);
    for (const file of req.files) { fs.unlink(file.path, (err) => { }); }
    return res.send({ message: 'PDFs processed successfully' });
  } catch (error) {
    console.error("Error in /upload_files:", error.message);
    return res.status(500).send({ error: 'Internal Server Error' });
  }
});
router.post('/ask_question', async function (req, res, next) {
  try {
    const { question } = req.body;
    let result = await geminiService.askQuestions(question, req.query);
    result.response = result.response.replace(/```json/g, '').replace(/```/g, '').trim();
    return res.send(result);
  } catch (error) {
    console.error("Error in /ask_question:", error.message);
    return res.status(500).send({ error: 'Internal Server Error' });
  }
});
module.exports = router;
