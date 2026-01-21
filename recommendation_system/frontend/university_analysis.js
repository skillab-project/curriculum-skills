// =============================
// Base API URL for all requests
// =============================
const API_BASE_URL = "http://localhost:8000";

// =============================
// Function to update metrics on the dashboard
// =============================
function updateMetrics(metrics) {
    // Update total study programs metric
    document.getElementById("total-programs").textContent =
        metrics.total_programs ?? "0";

    // Update recognized skills (ESCO) metric
    document.getElementById("recognized-skills").textContent =
        metrics.recognized_skills ?? "0";
}

// =============================
// Function to load universities into the custom dropdown
// =============================
async function loadUniversitiesCustom() {
    const wrapper = document.querySelector('.custom-select-wrapper'); // Wrapper div for the custom select
    const trigger = wrapper.querySelector('.custom-select-trigger'); // Dropdown trigger button
    const optionsContainer = wrapper.querySelector('.custom-options'); // Container for dropdown options

    // Fetch the list of universities from the API
    try {
        const response = await fetch(`${API_BASE_URL}/universities`);
        const universities = await response.json();

        // Clear previous options
        optionsContainer.innerHTML = '';

        // Populate dropdown with universities
        universities.forEach(univ => {
            const option = document.createElement('div');
            option.classList.add('custom-option');
            option.dataset.value = univ.university_id; // Store university ID in data attribute
            option.textContent = `${univ.university_name} (${univ.country})`; // Display name + country
            optionsContainer.appendChild(option);
        });
    } catch (err) {
        // Display error message if fetching fails
        optionsContainer.innerHTML = '<div class="custom-option">Failed to load universities</div>';
        console.error(err);
    }

    // =============================
    // Dropdown toggle logic
    // =============================
    trigger.addEventListener('click', () => {
        wrapper.querySelector('.custom-select').classList.toggle('open'); // Show/hide options
    });

    // =============================
    // Handle option selection
    // =============================
    optionsContainer.addEventListener('click', e => {
        if (e.target.classList.contains('custom-option')) {
            // Remove previous selection highlight
            const selected = optionsContainer.querySelector('.selected');
            if (selected) selected.classList.remove('selected');

            // Highlight the clicked option
            e.target.classList.add('selected');

            // Update the trigger text to show selected university
            trigger.textContent = e.target.textContent;

            // Close the dropdown
            wrapper.querySelector('.custom-select').classList.remove('open');

            // Enable action buttons once a university is selected
            const selectedUnivId = e.target.dataset.value;
            document.getElementById('degrees-btn').disabled = false;
            document.getElementById('courses-btn').disabled = false;

            // Fetch and display metrics for the selected university
            fetch(`${API_BASE_URL}/metrics/${selectedUnivId}`)
                .then(res => res.json())
                .then(metrics => updateMetrics(metrics))
                .catch(err => {
                    console.error(err);
                    // Display error if metrics fetch fails
                    updateMetrics({ total_programs: 'Error', recognized_skills: 'Error' });
                });
        }
    });

    // =============================
    // Close dropdown if clicking outside of it
    // =============================
    document.addEventListener('click', e => {
        if (!wrapper.contains(e.target)) {
            wrapper.querySelector('.custom-select').classList.remove('open');
        }
    });
}

// =============================
// Navigation buttons for running analyses
// =============================

// New Degree Analysis button
document.getElementById('degrees-btn').addEventListener('click', () => {
    const selected = document.querySelector('.custom-option.selected');
    if (!selected) return alert('Please select a university first.');
    const univ_id = selected.dataset.value;

    // Navigate to new degree proposals page with selected university ID
    window.location.href = `new_degree_purposals.html?type=degrees&univ_id=${univ_id}`;
});

// Course Gap Analysis button
document.getElementById('courses-btn').addEventListener('click', () => {
    const selected = document.querySelector('.custom-option.selected');
    if (!selected) return alert('Please select a university first.');
    const univ_id = selected.dataset.value;

    // Navigate to curriculum enhancement page with selected university ID
    window.location.href = `curriculum_enhancement.html?type=courses&univ_id=${univ_id}`;
});

// =============================
// Initialize dropdown and load universities when DOM is ready
// =============================
document.addEventListener('DOMContentLoaded', loadUniversitiesCustom);
