// Base URL of the backend API
const API_BASE_URL = 'http://localhost:8000';

/**
 * Converts a normalized score (0–1) into a green color gradient.
 */
function scoreToCourseColor(score) {
    const clampedScore = Math.max(0, Math.min(1, score));
    const lowR = 198, lowG = 226, lowB = 189;  // Light green
    const highR = 40, highG = 167, highB = 69; // Dark green
    const r = Math.round(lowR + (highR - lowR) * clampedScore);
    const g = Math.round(lowG + (highG - lowG) * clampedScore);
    const b = Math.round(lowB + (highB - lowB) * clampedScore);
    return `rgb(${r}, ${g}, ${b})`;
}

/**
 * Creates a horizontal stacked bar chart for skills per course
 */
function displayCourseHeatmap(courses, topSkillsLimit = 5) {
    let validCourses = courses.filter(c => c.new_skills || c.compatible_skills);

    const maxCoursesToDisplay = 5;
    if (validCourses.length > maxCoursesToDisplay) {
        validCourses = validCourses.slice(0, maxCoursesToDisplay);
    }

    if (!validCourses.length) {
        const canvas = document.getElementById('skillsHeatmapChart');
        if (canvas) canvas.style.display = 'none';
        return;
    }

    const labels = validCourses.map(c => c.course_name || 'Unknown Course');
    const compatibleSkillsCount = validCourses.map(c => (c.compatible_skills || []).length);
    const newSkillsCount = validCourses.map(c => (c.new_skills || []).length);
    const allSkillsCount = compatibleSkillsCount.map((count, i) => count + newSkillsCount[i]);
    const maxSkillsForAxis = Math.max(...allSkillsCount, 1);

    const canvas = document.getElementById('skillsHeatmapChart');
    if (canvas) {
        canvas.style.height = `${(validCourses.length * 35) + 150}px`;
        canvas.style.display = 'block';
    }

    const ctx = canvas ? canvas.getContext('2d') : null;
    if (!ctx) return;

    const datasets = [
        {
            label: 'Compatible Skills (Existing)',
            data: compatibleSkillsCount,
            backgroundColor: 'rgba(13,202,240,0.8)',
            borderColor: '#0dcaf0',
            borderWidth: 1.2,
            borderRadius: 5,
            barThickness: 25,
        },
        {
            label: 'New Skills (Enhancement)',
            data: newSkillsCount,
            backgroundColor: 'rgba(40, 167, 69, 0.9)',
            borderColor: '#146c43',
            borderWidth: 1.2,
            borderRadius: 5,
            barThickness: 25,
        }
    ];

    if (window.skillsHeatmapChart && typeof window.skillsHeatmapChart.destroy === 'function') {
        window.skillsHeatmapChart.destroy();
    }

    document.querySelector('.chart-title').textContent =
        ` Skills Heatmap per Course (Top ${validCourses.length} Courses Breakdown)`;

    window.skillsHeatmapChart = new Chart(ctx, {
        type: 'bar',
        data: { labels, datasets },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: { boxWidth: 14, usePointStyle: true, pointStyle: 'rectRounded' }
                },
                tooltip: {
                    callbacks: {
                        label: context => `${context.dataset.label}: ${context.raw} skills`,
                        title: context => {
                            const courseIndex = context[0].dataIndex;
                            return `${context[0].label} (Total Skills: ${allSkillsCount[courseIndex]})`;
                        },
                        footer: context => {
                            const course = validCourses[context[0].dataIndex];
                            let footer = '';
                            if ((course.new_skills || []).length)
                                footer += `New: ${course.new_skills.join(', ')}\n`;
                            if ((course.compatible_skills || []).length)
                                footer += `Compatible: ${course.compatible_skills.join(', ')}`;
                            return footer.trim();
                        }
                    }
                }
            },
            scales: {
                x: {
                    stacked: true,
                    title: { display: true, text: `Count of Skills` },
                    min: 0,
                    max: Math.ceil(maxSkillsForAxis / 5) * 5 || 5,
                    ticks: { stepSize: 1, font: { size: 12 } }
                },
                y: {
                    stacked: true,
                    ticks: { font: { size: 14, weight: 'bold' } }
                }
            }
        }
    });
}

/**
 * Renders recommended courses
 */
function displayCourseRecommendations(courses, degreeName) {
    const resultsContainer = document.getElementById('course-recommendation-list');
    const titleElement = document.getElementById('courses-title');

    titleElement.textContent = ` Recommended Courses for: ${decodeURIComponent(degreeName)}`;

    if (!courses || courses.length === 0) {
        resultsContainer.innerHTML =
            `<li class="course-card" style="border-left-color: #dc3545;">❌ No recommended courses found.</li>`;
        const canvas = document.getElementById('skillsHeatmapChart');
        if (canvas) canvas.style.display = 'none';
        return;
    }

    let htmlContent = '';
    courses.forEach(course => {
        const score = course.score ? course.score.toFixed(3) : 'N/A';
        const color = scoreToCourseColor(course.score || 0);

        htmlContent += `
            <li class="course-card" style="border-left-color: ${color};">
                <div class="card-header">
                    <p class="course-name">${course.course_name || 'Unknown Course'}</p>
                    <div class="score-badge" style="background-color: ${color};">Score: ${score}</div>
                </div>
                <div class="info-grid mt-3">
                    <div class="info-item"><h6>🎯 Objectives</h6>
                        <p>${(course.objectives || 'N/A').substring(0, 150)}...</p></div>
                    <div class="info-item"><h6>💡 Learning Outcomes</h6>
                        <p>${(course.learning_outcomes || 'N/A').substring(0, 150)}...</p></div>
                </div>
                <div class="info-section mt-3 p-3" style="border: 1px dashed #ced4da; background-color: #f7f7f7;">
                    <h6>🌐 Description Summary</h6>
                    <p style="font-size: 0.85em;">${(course.description || 'N/A').substring(0, 200)}...</p>
                    <details>
                        <summary style="cursor:pointer;color:#007bff;font-weight:500;">Full Details</summary>
                        <h6>Full Description</h6><p>${course.description || 'N/A'}</p>
                        <h6>Course Content</h6><p>${course.course_content || 'N/A'}</p>
                    </details>
                </div>
            </li>`;
    });

    resultsContainer.innerHTML = htmlContent;
    displayCourseHeatmap(courses);
}

/**
 * Load degree programs
 */
async function loadDegreePrograms(univId) {
    const dropdown = document.getElementById('degree-dropdown');
    dropdown.innerHTML = '<option value="">Select Degree Program</option>';

    try {
        const response = await fetch(`${API_BASE_URL}/universities/${univId}/degrees`);
        if (!response.ok) throw new Error('Failed to fetch degrees');

        const degrees = await response.json();
        degrees.forEach(degree => {
            const option = document.createElement('option');
            option.value = encodeURIComponent(degree.degree_titles?.[0] || degree.degree_type);
            option.textContent = `${degree.degree_type}: ${degree.degree_titles?.join(', ') || ''}`;
            dropdown.appendChild(option);
        });
    } catch (error) {
        console.error("Error loading degrees:", error);
        const option = document.createElement('option');
        option.textContent = "Failed to load degrees";
        dropdown.appendChild(option);
    }
}

/**
 * Triggered when the user selects a degree
 */
document.getElementById('degree-dropdown').addEventListener('change', (event) => {
    const degreeName = event.target.value;
    const params = new URLSearchParams(window.location.search);
    const univId = params.get('univ_id');
    if (degreeName && univId) {
        fetchAndDisplayRecommendations(univId, degreeName);
    }
});

/**
 * Fetch recommended courses
 */
async function fetchAndDisplayRecommendations(univId, degreeName) {
    const listElement = document.getElementById('course-recommendation-list');

    const endpoint = `${API_BASE_URL}/recommend/courses/${univId}/${degreeName}`;
    try {
        const response = await fetch(endpoint);
        if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);

        const data = await response.json();
        displayCourseRecommendations(data.recommendations || [], degreeName);

    } catch (error) {
        console.error("Error loading courses:", error);
        listElement.innerHTML =
            `<li class="course-card" style="border-left-color: #dc3545;">Failed to load data: ${error.message}</li>`;
        const canvas = document.getElementById('skillsHeatmapChart');
        if (canvas) canvas.style.display = 'none';
    }
}

/**
 * Initialize page
 */
async function initPage() {
    const params = new URLSearchParams(window.location.search);
    const univId = params.get('univ_id');
    if (!univId) return;

    let univName = `University ID: ${univId}`;
    try {
        const response = await fetch(`${API_BASE_URL}/universities`);
        if (response.ok) {
            const universities = await response.json();
            const targetUniv = universities.find(u => String(u.university_id) === univId);
            if (targetUniv) univName = targetUniv.university_name;
        }
    } catch (error) {
        console.warn("Could not fetch university name:", error);
    }

    document.getElementById('courses-header').textContent = `University: ${univName}`;
    await loadDegreePrograms(univId);
}

initPage();
