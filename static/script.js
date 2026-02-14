document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const formData = new FormData(e.target);
    const data = Object.fromEntries(formData.entries());

    // Convert numeric fields
    data.tenure = parseInt(data.tenure);
    data.MonthlyCharges = parseFloat(data.MonthlyCharges);
    data.TotalCharges = parseFloat(data.TotalCharges);
    data.SeniorCitizen = parseInt(data.SeniorCitizen);

    // Show loading state
    const btn = e.target.querySelector('button');
    const originalText = btn.innerText;
    btn.innerText = 'Analyzing...';
    btn.disabled = true;

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });

        const result = await response.json();

        // Update UI
        const resultContainer = document.getElementById('resultContainer');
        const churnResult = document.getElementById('churnResult');
        const probBar = document.getElementById('probBar');
        const probText = document.getElementById('probText');

        resultContainer.classList.remove('hidden');
        churnResult.innerText = result.churn === 'Yes' ? 'HIGH RISK' : 'LOW RISK';
        churnResult.className = 'churn-status ' + (result.churn === 'Yes' ? 'status-high' : 'status-low');

        const probPercent = (result.probability * 100).toFixed(1);
        probBar.style.width = probPercent + '%';
        probText.innerText = `Churn Probability: ${probPercent}%`;

        // Smooth scroll to result
        resultContainer.scrollIntoView({ behavior: 'smooth' });

    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred during prediction.');
    } finally {
        btn.innerText = originalText;
        btn.disabled = false;
    }
});

// Simple nav highlighting
window.addEventListener('scroll', () => {
    const sections = document.querySelectorAll('section');
    const navItems = document.querySelectorAll('.nav-item');

    let current = '';
    sections.forEach(section => {
        const sectionTop = section.offsetTop;
        if (pageYOffset >= sectionTop - 100) {
            current = section.getAttribute('id');
        }
    });

    navItems.forEach(item => {
        item.classList.remove('active');
        if (item.getAttribute('href').substring(1) === current) {
            item.classList.add('active');
        }
    });
});
