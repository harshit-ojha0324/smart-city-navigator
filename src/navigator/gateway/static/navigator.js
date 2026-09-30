(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const form = $('planner-form');
  const buttons = [...document.querySelectorAll('[data-mode]')];
  const labels = {trip: 'Find my route', status: 'Check service', question: 'Find an answer'};
  const titles = {trip: 'Your trip details', status: 'Your service report', question: 'Here’s what we found'};
  const nodeLabels = {understand: 'Understanding your request', supervisor: 'Choosing the next lookup', route_planner: 'Planning your route', service_advisor: 'Checking service updates', station_info: 'Looking up your station', synthesize: 'Putting it all together'};
  let mode = 'trip', stream = null, timer = null, sequence = 0, lastQuestion = '';

  function stop() {
    sequence += 1;
    if (stream) stream.close();
    stream = null;
    clearTimeout(timer);
    $('submit').disabled = false;
    $('submit-label').textContent = labels[mode];
    $('cancel').hidden = true;
    $('result').setAttribute('aria-busy', 'false');
  }

  function changeMode(next) {
    stop();
    mode = next;
    buttons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.mode === mode)));
    ['trip', 'status', 'question'].forEach(name => {
      const panel = $(name + '-fields');
      panel.hidden = name !== mode;
      panel.querySelectorAll('input, select, textarea, button').forEach(field => { field.disabled = name !== mode; });
    });
    $('submit-label').textContent = labels[mode];
    $('suggestions').hidden = mode !== 'trip';
    $('result').hidden = true;
  }

  function fail(message) {
    stop();
    $('result').dataset.state = 'error';
    $('result-title').textContent = 'We couldn’t finish that lookup.';
    $('progress').textContent = '';
    $('error').textContent = message;
    $('error').hidden = false;
    $('retry').hidden = false;
  }

  function ask(question) {
    stop();
    lastQuestion = question;
    const current = sequence;
    $('result').hidden = false;
    $('result').scrollIntoView({behavior: 'auto', block: 'nearest'});
    $('result').dataset.state = 'loading';
    $('result').dataset.simulated = 'false';
    $('result').setAttribute('aria-busy', 'true');
    $('result-label').textContent = mode === 'trip' ? 'YOUR JOURNEY' : mode === 'status' ? 'SERVICE CHECK' : 'YOUR ANSWER';
    $('result-title').textContent = 'Finding your way…';
    $('answer').textContent = '';
    $('error').hidden = true;
    $('retry').hidden = true;
    $('steps').replaceChildren();
    $('trace').hidden = true;
    $('trace').open = false;
    $('progress').textContent = 'Connecting to the transit planner…';
    $('submit').disabled = true;
    $('submit-label').textContent = 'Working on it…';
    $('cancel').hidden = false;
    timer = setTimeout(() => {
      if (current === sequence) fail('This is taking longer than expected. Please try again in a moment.');
    }, 90000);
    try {
      stream = new EventSource('/api/stream?' + new URLSearchParams({q: question}));
    } catch (_) {
      fail('Unable to connect. Please check your connection and try again.');
      return;
    }
    stream.onmessage = event => {
      if (current !== sequence) return;
      let data;
      try { data = JSON.parse(event.data); } catch (_) {
        fail('We received an incomplete response. Please try again.');
        return;
      }
      if (data.node === 'error') {
        fail('The transit planner is unavailable right now. Please try again shortly.');
        return;
      }
      if (data.node === 'done') {
        if (typeof data.answer !== 'string' || !data.answer.trim()) {
          fail('No answer was returned. Try a more specific station or line.');
          return;
        }
        stop();
        $('result').dataset.state = 'complete';
        $('result').dataset.simulated = String(/simulated/i.test(data.answer));
        $('result-title').textContent = titles[mode];
        $('progress').textContent = '';
        // All user/tool/model content is text, never interpreted as markup.
        $('answer').textContent = data.answer;
        return;
      }
      $('progress').textContent = (nodeLabels[data.node] || 'Checking your request') + '…';
      if (typeof data.text === 'string' && data.text) {
        const item = document.createElement('li');
        item.textContent = data.text;
        $('steps').append(item);
        $('trace').hidden = false;
      }
    };
    stream.onerror = () => {
      if (current === sequence) fail('The connection was interrupted. Please try again.');
    };
  }

  form.addEventListener('submit', event => {
    event.preventDefault();
    let question;
    if (mode === 'trip') {
      const origin = $('origin').value.trim(), destination = $('destination').value.trim();
      if (!origin || !destination) {
        (origin ? $('destination') : $('origin')).focus();
        return;
      }
      question = `How do I get from ${origin} to ${destination}` + ($('include-status').checked ? ' and are there delays?' : '?');
    } else if (mode === 'status') {
      question = $('line').value === 'all' ? 'What is the subway service status?' : `Is the ${$('line').value} train running?`;
    } else {
      question = $('question').value.trim();
      if (!question) { $('question').focus(); return; }
    }
    ask(question);
  });
  buttons.forEach(button => button.addEventListener('click', () => changeMode(button.dataset.mode)));
  $('swap').addEventListener('click', () => {
    const original = $('origin').value;
    $('origin').value = $('destination').value;
    $('destination').value = original;
  });
  document.querySelectorAll('[data-origin]').forEach(button => button.addEventListener('click', () => {
    changeMode('trip');
    $('origin').value = button.dataset.origin;
    $('destination').value = button.dataset.destination;
    $('submit').focus();
  }));
  $('cancel').addEventListener('click', () => {
    stop();
    $('result').hidden = true;
    $('submit').focus();
  });
  $('retry').addEventListener('click', () => ask(lastQuestion));
  window.addEventListener('pagehide', stop);
})();
