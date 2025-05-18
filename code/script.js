document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('search-form');
  const resultsSection = document.getElementById('results');

  form.addEventListener('submit', (e) => {
    e.preventDefault();

    const query = document.getElementById('search-box').value;
    const confidence = document.getElementById('confidence').value;
    const maxResults = document.getElementById('max-results').value;

    // Placeholder for backend integration
    // Replace this with actual API call to your backend
    // Example:
    // fetch(`/api/search?q=${query}&confidence=${confidence}&max=${maxResults}`)
    //   .then(response => response.json())
    //   .then(data => displayResults(data));

    // For demonstration, we'll use mock data
    const mockData = [
      {
        title: 'Document 1',
        snippet: 'Sample text containing the keyword...',
        confidence: 75
      },
      {
        title: 'Document 2',
        snippet: 'Another example with the searched term...',
        confidence: 80
      }
    ];

    displayResults(mockData);
  });

  function displayResults(data) {
    resultsSection.innerHTML = '';

    if (data.length === 0) {
      resultsSection.innerHTML = '<p>No results found.</p>';
      return;
    }

    data.forEach(item => {
      const div = document.createElement('div');
      div.classList.add('result-item');
      div.innerHTML = `
        <h3>${item.title}</h3>
        <p>${item.snippet}</p>
        <p><strong>Confidence:</strong> ${item.confidence}%</p>
      `;
      resultsSection.appendChild(div);
    });
  }
});
