document.getElementById('chatForm').addEventListener('submit', function(e) {
    e.preventDefault();
    const inputField = document.getElementById('userInput');
    const messageText = inputField.value.trim();
    
    if (messageText === '') return;

    //Instantly display user's message in the stream
    renderMessage(messageText, 'user-message');
    inputField.value = '';

    // Create a temporary placeholder for the bot's response
    const botMessageDiv = renderPlaceholder();

    // Send the message payload to Flask backend
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message: messageText })
    })
    .then(response => {
        if (!response.ok) throw new Error('Network response was not ok');
        return response.json();
    })
    .then(data => {
        // Update the placeholder with the real answer from model.py
        updateBotResponse(botMessageDiv, data.answer);
    })
    .catch(error => {
        console.error('Error:', error);
        updateBotResponse(botMessageDiv, "Sorry, I encountered an error while retrieving that data. Please try again.");
    });
});

// Helper to trigger chip selections directly
function sendPreset(text) {
    document.getElementById('userInput').value = text;
    document.getElementById('chatForm').dispatchEvent(new Event('submit'));
}

// Renders the user text bubble
function renderMessage(text, className) {
    const chatMessages = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${className}`;
    messageDiv.innerHTML = `<div class="message-content"><p>${text}</p></div>`;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Renders a loading/thinking state container
function renderPlaceholder() {
    const chatMessages = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot-message';
    messageDiv.innerHTML = `<div class="message-content"><p><em>Analyzing financial vectors...</em></p></div>`;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return messageDiv;
}

// Replaces the placeholder text with actual LLM output
function updateBotResponse(element, answer) {
    element.innerHTML = `
        <div class="message-content">
            <p>${answer}</p>
        </div>
    `;
    const chatMessages = document.getElementById('chatMessages');
    chatMessages.scrollTop = chatMessages.scrollHeight;
}