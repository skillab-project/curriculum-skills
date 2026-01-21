

// Base URL for the API
const API_BASE_URL = 'http://localhost:8000';


// =======================================================
// 1. Helper Functions
// =======================================================

/**
 * Convert a score (0–1) into a pastel green color for bars.
 * @param {number} score - A number between 0 and 1
 * @returns {string} - CSS rgb color string
 */
function scoreToColor(score) {
    const clampedScore = Math.max(0, Math.min(1, score)); // Clamp score between 0 and 1
    const lowR = 223, lowG = 246, lowB = 228; // Light green RGB
    const highR = 91, highG = 184, highB = 92; // Dark green RGB

    const r = Math.round(lowR + (highR - lowR) * clampedScore);
    const g = Math.round(lowG + (highG - lowG) * clampedScore);
    const b = Math.round(lowB + (highB - lowB) * clampedScore);

    return `rgb(${r}, ${g}, ${b})`;
}

/**
 * Generate HTML for a top skills heatmap
 * @param {Array} topSkills - Array of skills with name and score
 * @returns {string} - HTML string
 */
function generateSkillsHeatmap(topSkills) {
    if (!topSkills || topSkills.length === 0) {
        return `<p style="color: #6c757d; font-size: 0.9em; margin-top: 5px;">
                  No associated skills found.</p>`;
    }

    // Add a "Top Skills" section
    let html = `
        <div class="heatmap-section-full">
            <h5><i class="fas fa-lightbulb"></i> Top Skills</h5>
    `;

    // Generate a bar for each skill (max 5 skills)
    html += topSkills.slice(0, 5).map(skill => {
        const skillScore = skill.skill_score || 0;
        const width = Math.round(skillScore * 100);
        const barColor = scoreToColor(skillScore);

        return `
            <div class="skill-bar">
                <p><strong>${skill.skill_name}</strong> 
                   <span style="font-weight: 600; color: ${barColor};">${width}%</span>
                </p>
                <div class="bar-wrap">
                    <div class="bar" style="width: ${width}%; background-color: ${barColor};"></div>
                </div>
            </div>
        `;
    }).join('');

    html += `</div>`; // close heatmap section
    return html;
}

/**
 * Generate HTML for degree metrics bars
 * @param {Object} metrics - Object containing metrics values
 * @returns {string} - HTML string
 */
function generateMetricsBars(metrics) {
    if (!metrics) return '';

    return `
        <div class="heatmap-section-full">
            <h5>Degree Metrics</h5>

            <div class="metric-item">
                <p>
                    <span style="font-weight:700;">Frequency</span>
                    <span style="font-weight:700;">${metrics.frequency}%</span>
                </p>
                <div class="metric-bar-wrap">
                    <div class="metric-bar" style="width:${metrics.frequency}%; background-color: var(--analyst-color);"></div>
                </div>
            </div>

            <div class="metric-item">
                <p>
                    <span style="font-weight:700;">Compatibility</span>
                    <span style="font-weight:700;">${metrics.compatibility}%</span>
                </p>
                <div class="metric-bar-wrap">
                    <div class="metric-bar" style="width:${metrics.compatibility}%; background-color: var(--primary-color);"></div>
                </div>
            </div>

            <div class="metric-item">
                <p>
                    <span style="font-weight:700;">Novelty</span>
                    <span style="font-weight:700;">${metrics.novelty}%</span>
                </p>
                <div class="metric-bar-wrap">
                    <div class="metric-bar" style="width:${metrics.novelty}%; background-color: var(--info-blue-color);"></div>
                </div>
            </div>

            <div class="metric-item">
                <p>
                    <span style="font-weight:700;">Skill Enrichment</span>
                    <span style="font-weight:700;">${metrics.skill_enrichment}</span>
                </p>
                <div class="metric-bar-wrap">
                    <div class="metric-bar" style="width:${Math.min(metrics.skill_enrichment * 20, 100)}%; background-color: var(--pastel-green);"></div>
                </div>
            </div>
        </div>
    `;
}



// =======================================================
// 1. NORMALIZATION + DEDUPE (ΝΕΟ - ΒΑΛΤΟ ΠΑΝΩ ΠΑΝΩ)
// =======================================================

function normalizeRec(rec) {
    return {
        name: rec.degree?.trim(),      // 🔥 μοναδικό πεδίο
        degree_type: rec.degree_type || 'BSc/BA',
        score: rec.score || 0,
        top_skills: rec.top_skills || [],
        metrics: rec.metrics || {}
    };
}

function dedupeRecs(list) {
    const map = new Map();
    list.forEach(rec => {
        const clean = normalizeRec(rec);
        if (clean.name && !map.has(clean.name)) { // 🔥 μόνο βάση του name
            map.set(clean.name, clean);
        }
    });
    return Array.from(map.values());
}



// =======================================================
// 3. Display Recommendations
// =======================================================

/**
 * Render recommendation cards in the DOM
 * @param {Array} recommendations - List of degree/course recommendations
 * @param {string} type - 'degrees' or 'courses'
 * @param {string} univId - University ID
 */
function displayRecommendations(recommendations, type, univId) {
    const resultsContainer = document.getElementById('recommendation-list');
    const titleElement = document.getElementById('results-title');
    const loadingSpinner = document.getElementById('loading-spinner');

    loadingSpinner.style.display = 'none';

    // No results case
    if (!recommendations || !Array.isArray(recommendations) || recommendations.length === 0) {
        titleElement.textContent = `📊 No Results Found`;
        resultsContainer.innerHTML = `<li style="color: #dc3545; padding: 20px; background: #fff; border-left: 8px solid #dc3545; font-size: 1.1em;">
            ❌ No new ${type === 'degrees' ? 'degree' : 'course'} proposals found for this university.
        </li>`;
        return;
    }

    // Build HTML for each recommendation
    const htmlContent = recommendations.map((rec) => {
        const itemName = rec.name;   // 🔥 πάντα 1 όνομα, πάντα καθαρό




        const scorePercent = rec.score != null ? Math.round(Math.max(1, Math.min(rec.score * 100, 100))) : '—';
        const itemColor = scoreToColor((rec.score || 0));

        const degreeType = rec.degree_type || 'BSc/BA';

        const compatibilityPercent = rec.metrics?.compatibility != null ? Math.round(Math.min(Math.max(rec.metrics.compatibility * 100, 0), 100)) : '—';
        const noveltyPercent = rec.metrics?.novelty != null ? Math.round(Math.min(Math.max(rec.metrics.novelty * 100, 0), 100)) : '—';
        const frequencyPercent = rec.metrics?.frequency != null ? Math.round(Math.min(Math.max(rec.metrics.frequency * 100, 0), 100)) : '—';
        const skillEnrichment = rec.metrics?.skill_enrichment != null ? rec.metrics.skill_enrichment : '—';

        // Generate heatmap and metric bars
        const skillsHtml = generateSkillsHeatmap(rec.top_skills) + generateMetricsBars({
            frequency: frequencyPercent,
            compatibility: compatibilityPercent,
            novelty: noveltyPercent,
            skill_enrichment: skillEnrichment
        });

        return `
            <li class="recommendation-item recommendation-card" style="border-left-color: ${itemColor};">
                <div class="card-header">
                    <div class="degree-info">
                        <h4 class="degree-name">${itemName} <span class="degree-type">[${degreeType}]</span></h4>
                    </div>
                    <div class="score-badge" style="background-color: ${itemColor};">
                        ${type === 'courses' ? 'Proposal' : `Score: ${scorePercent}%`}
                    </div>
                </div>
                <div class="card-content-full-width">
                    ${skillsHtml}
                    <div class="action-section-centered">
                        <button class="recommend-courses-btn green-btn" 
                                data-degree-name="${itemName}"
                                data-univ-id="${univId}">
                            <i class="fas fa-tasks"></i> Generate Course Recommendations
                        </button>
                    </div>
                </div>
            </li>
        `;
    }).join('');

    resultsContainer.innerHTML = htmlContent;

    // Add click event listeners to all "Generate Course Recommendations" buttons
    document.querySelectorAll('.recommend-courses-btn').forEach(button => {
        button.addEventListener('click', handleRecommendCoursesClick);
    });
}


// =======================================================
// 4. Main Function: Load Recommendations
// =======================================================

/**
 * Load recommendations from API and display them
 */
async function loadRecommendations() {
    const params = new URLSearchParams(window.location.search);
    const univId = params.get('univ_id');
    const type = params.get('type');

    const infoElement = document.getElementById('university-info');
    const titleElement = document.getElementById('results-title');
    const loadingSpinner = document.getElementById('loading-spinner');

    loadingSpinner.style.display = 'block';

    // Validate URL parameters
    if (!univId || !type) {
        infoElement.innerHTML = `<i class="fas fa-exclamation-triangle"></i> Error: University ID or Recommendation Type missing.`;
        loadingSpinner.style.display = 'none';
        return;
    }

    // Attempt to fetch university name for display
    let univName = `University ID: ${univId}`;
    try {
        const univsResponse = await fetch(`${API_BASE_URL}/universities`);
        const universities = await univsResponse.json();
        const selectedUniv = universities.find(u => String(u.university_id) === univId);
        if (selectedUniv) univName = `${selectedUniv.university_name} (${selectedUniv.country})`;
    } catch (e) {
        console.error("Could not fetch university name:", e);
    }

    infoElement.innerHTML = `<i class="fas fa-info-circle"></i> Find the best programs and courses based on your profile.`;
    titleElement.textContent = `Recommended Degrees for: ${univName}`;

    // Determine API endpoint based on recommendation type
    let endpoint = '';
    if (type === 'degrees') endpoint = `${API_BASE_URL}/recommend/degrees/${univId}`;
    else if (type === 'courses') endpoint = `${API_BASE_URL}/recommendations/university/${univId}`;
    else {
        infoElement.textContent = "Unknown recommendation type.";
        loadingSpinner.style.display = 'none';
        return;
    }

    try {
        const response = await fetch(endpoint);
        if (!response.ok) {
            let errorDetail = await response.text();
            try { errorDetail = JSON.parse(errorDetail).detail || errorDetail; } catch (e) { }
            throw new Error(`HTTP ${response.status}: ${errorDetail}`);
        }

        const data = await response.json();
        let recommendations = [];
        if (type === 'degrees') recommendations = dedupeRecs(data.recommended_degrees || []);
        else if (type === 'courses') recommendations = dedupeRecs(data.recommendations?.new_degree_proposals || []);

        displayRecommendations(recommendations, type, univId);




    } catch (error) {
        console.error(`Error loading ${type} recommendations:`, error);
        loadingSpinner.style.display = 'none';
        titleElement.textContent = `❌ Error Loading Results`;
        document.getElementById('recommendation-list').innerHTML =
            `<li style="color: #dc3545; padding: 20px; background: #fff; border-left: 8px solid #dc3545; font-size: 1.1em;">
                <i class="fas fa-server"></i> Failed to load data: ${error.message}. Ensure FastAPI server is running.
            </li>`;
    } finally {
        loadingSpinner.style.display = 'none';
    }
    // After main recommendations load:
    displayPastRecommendations();

}

// Initialize recommendations on DOM content loaded
document.addEventListener('DOMContentLoaded', loadRecommendations);

function displayPastRecommendations() {
    const viewed = getViewedDegrees();
    const container = document.getElementById("past-recommendations");

    // --- Πάρε τις τρέχουσες recommendations από το DOM ---
    let criteriaRecommendations = [];
    document.querySelectorAll('.recommendation-item.recommendation-card:not(.past-card)').forEach(card => {
        const degreeName = card.querySelector('.degree-name')?.childNodes[0]?.textContent.trim();
        const univId = new URLSearchParams(window.location.search).get("univ_id");
        if (degreeName) criteriaRecommendations.push({ degreeName, univId });
    });

    // --- Δημιουργία key set για τρέχοντα recommendations ---
    // --- Δημιουργία key set για τρέχοντα recommendations ---
    const criteriaKeySet = new Set(
        criteriaRecommendations.map(r => r.degreeName) // μόνο το όνομα
    );

    // --- Φιλτράρισμα ιστορικού ώστε να μην εμφανίζονται πτυχία που είναι ήδη στα τρέχοντα ---
    const filteredViewed = viewed.filter(v =>
        !criteriaKeySet.has(v.degreeName) // μόνο βάση name
    );


    // Αν δεν υπάρχουν παλαιά recommendations
    if (filteredViewed.length === 0) {
        container.innerHTML = "<li>No history yet.</li>";
        return;
    }

    // Render ΜΟΝΟ τα filteredViewed
    container.innerHTML = filteredViewed.map(v => `
        <li class="recommendation-item recommendation-card past-card" 
            data-degree-name="${v.degreeName}" data-univ-id="${v.univId}" style="cursor:pointer;">
            <h4>${v.degreeName}</h4>
        </li>
    `).join("");

    // Click listener για κάθε παλαιό viewed πρόγραμμα
    document.querySelectorAll('.past-card').forEach(card => {
        card.addEventListener('click', () => {
            const degreeName = card.getAttribute('data-degree-name');
            const univId = card.getAttribute('data-univ-id');
            window.location.href = `recommended_degree_plan.html?univ_id=${univId}&degree_name=${encodeURIComponent(degreeName)}`;
        });
    });

    setupCarousel();
}




/* =======================================================
   HISTORY FUNCTIONS (LocalStorage)
======================================================= */
function getViewedDegrees() {
    return JSON.parse(localStorage.getItem("degreeHistory")) || [];
}

function saveViewedDegree(degreeName, univId) {
    let history = JSON.parse(localStorage.getItem("degreeHistory")) || [];
    const entry = { degreeName: degreeName.trim(), univId };

    // Αποφυγή διπλών
    history = history.filter(h => !(h.degreeName === entry.degreeName && h.univId === entry.univId));
    history.push(entry);

    localStorage.setItem("degreeHistory", JSON.stringify(history));
}



async function handleRecommendCoursesClick(event) {
    const button = event.target.closest('button');
    const universityId = button.getAttribute('data-univ-id');
    const degreeName = button.getAttribute('data-degree-name');

    // 1) Save to local browser history (frontend only)
    saveViewedDegree(degreeName, universityId);


    // 2) Send interaction to the backend
    await fetch(`${API_BASE_URL}/interactions/add`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            user_id: 1,              // static user for now
            course_name: degreeName, // what user clicked
            interest_score: 1.0
        })
    });

    // 3) Navigate to degree recommendation page
    const encodedDegreeName = encodeURIComponent(degreeName);
    window.location.href =
        `recommended_degree_plan.html?univ_id=${universityId}&degree_name=${encodedDegreeName}`;

}


function setupCarousel() {
    const track = document.querySelector('.carousel-track');
    const leftBtn = document.querySelector('.carousel-btn.left');
    const rightBtn = document.querySelector('.carousel-btn.right');

    let currentIndex = 0;

    function updateCarousel() {
        const cards = track.querySelectorAll('li');
        if (!cards.length) return; // Αν δεν υπάρχουν κάρτες, σταματάμε

        const cardWidth = cards[0].offsetWidth;
        const gap = 20; // ίδιο με CSS gap
        const trackWidth = track.parentElement.offsetWidth;
        const visibleCount = Math.floor(trackWidth / (cardWidth + gap));
        const maxIndex = Math.max(cards.length - visibleCount, 0);

        // Περιορισμός currentIndex
        if (currentIndex > maxIndex) currentIndex = maxIndex;
        if (currentIndex < 0) currentIndex = 0;

        // Μετακίνηση
        track.style.transform = `translateX(-${currentIndex * (cardWidth + gap)}px)`;

        // Απενεργοποίηση κουμπιών αν φτάσαμε άκρα
        leftBtn.disabled = currentIndex === 0;
        rightBtn.disabled = currentIndex === maxIndex;
    }

    leftBtn.addEventListener('click', () => {
        currentIndex--;
        updateCarousel();
    });

    rightBtn.addEventListener('click', () => {
        currentIndex++;
        updateCarousel();
    });

    window.addEventListener('resize', updateCarousel);
    updateCarousel();
}