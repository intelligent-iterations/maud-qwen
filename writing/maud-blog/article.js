'use strict';
const results = JSON.parse(document.querySelector('#experiment-data').textContent);
const definitions = {
  question_macro_f1: 'Our primary metric: calculate F1 across every valid answer label, then average equally across the 119 eligible question/subquestion tasks. Higher is better.',
  accuracy: 'The fraction of all 5,283 test answers that match the expert label. Frequent answers contribute more examples to this metric. Higher is better.',
  rare_recall: 'Among the 154 test examples whose correct answer is rare in training, how many did each system answer correctly? Higher is better.'
};
const buttons = [...document.querySelectorAll('[data-metric]')];
function setMetric(metric) {
  buttons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.metric === metric)));
  document.querySelector('#metric-description').textContent = definitions[metric];
  document.querySelectorAll('[data-system]').forEach(row => {
    const value = results.systems[row.dataset.system][metric];
    row.querySelector('.bar-fill').style.setProperty('--value', `${value * 100}%`);
    row.querySelector('.bar-number').textContent = metric === 'question_macro_f1' ? value.toFixed(4) : `${(value * 100).toFixed(2)}%`;
  });
  document.querySelector('#axis-max').textContent = metric === 'question_macro_f1' ? '1.0' : '100%';
  document.querySelector('#axis-mid').textContent = metric === 'question_macro_f1' ? '0.5' : '50%';
}
buttons.forEach(button => button.addEventListener('click', () => setMetric(button.dataset.metric)));
const dialog = document.querySelector('#figure-dialog');
document.querySelectorAll('.plot-zoom').forEach(button => button.addEventListener('click', () => {
  const source = button.querySelector('img');
  document.querySelector('#detail-image').src = source.currentSrc || source.src;
  document.querySelector('#detail-image').alt = source.alt;
  document.querySelector('#detail-title').textContent = button.dataset.title;
  dialog.showModal();
}));
document.querySelector('#close-figure').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', event => {
  const box = dialog.getBoundingClientRect();
  if (event.target === dialog && (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom)) dialog.close();
});
window.addEventListener('beforeprint', () => setMetric('question_macro_f1'));
