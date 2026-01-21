// ======================================================
// Base API URL for backend endpoints
// ======================================================
const API_BASE = "http://localhost:8000";

// ======================================================
// Load dropdown filters for degree type, country, language
// ======================================================
async function loadFilters() {
    try {
        // Fetch degree types and populate dropdown
        const degrees = await fetch(`${API_BASE}/filters/degree-types`).then(r => r.json());
        degrees.forEach(d => document.getElementById("degreeType").appendChild(new Option(d, d)));

        // Fetch countries and populate dropdown
        const countries = await fetch(`${API_BASE}/filters/countries`).then(r => r.json());
        countries.forEach(c => document.getElementById("country").appendChild(new Option(c, c)));

        // Fetch languages and populate dropdown
        const languages = await fetch(`${API_BASE}/filters/languages`).then(r => r.json());
        languages.forEach(l => document.getElementById("language").appendChild(new Option(l, l)));
    } catch (err) {
        console.error("Error loading filters:", err);
    }
}

// ======================================================
// Load skills grouped by categories and create collapsible sections
// ======================================================
async function loadSkills() {
    try {
        const data = await fetch(`${API_BASE}/skills/grouped-by-categories`).then(r => r.json());
        const container = document.getElementById("skillsContainer");
        container.innerHTML = "";

        // Iterate over each skill category
        Object.keys(data).forEach(cat => {
            const catDiv = document.createElement("div");
            catDiv.className = "skill-category";

            // Category header with collapsible icon
            const header = document.createElement("div");
            header.className = "category-header";
            header.innerHTML = `<span>${cat.toUpperCase()}</span> <i class="fas fa-chevron-down"></i>`;
            catDiv.appendChild(header);

            // Container for skills within the category
            const content = document.createElement("div");
            content.className = "category-content";

            // Create checkbox for each skill
            data[cat].forEach(skill => {
                const label = document.createElement("label");
                label.innerHTML = `<input type="checkbox" value="${skill.id}" data-skill-name="${skill.name}"> ${skill.name}`;
                content.appendChild(label);
            });

            catDiv.appendChild(content);
            container.appendChild(catDiv);

            // Toggle visibility when header is clicked
            header.addEventListener("click", () => {
                const isVisible = content.style.display === "grid";
                content.style.display = isVisible ? "none" : "grid";
                header.classList.toggle("active", !isVisible);
                header.querySelector("i").classList.toggle("fa-chevron-down", isVisible);
                header.querySelector("i").classList.toggle("fa-chevron-up", !isVisible);
            });
        });
    } catch (err) {
        console.error("Error loading skills:", err);
    }
}

// ======================================================
// Create a circular progress indicator for recommendation scores
// ======================================================
function createCircularProgress(score) {
    const radius = 35;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference - (score / 100) * circumference;

    const svgNS = "http://www.w3.org/2000/svg";

    // Create SVG container
    const svg = document.createElementNS(svgNS, "svg");
    svg.setAttribute("width", "80");
    svg.setAttribute("height", "80");

    // Background circle
    const bg = document.createElementNS(svgNS, "circle");
    bg.setAttribute("cx", "40");
    bg.setAttribute("cy", "40");
    bg.setAttribute("r", radius);
    bg.setAttribute("class", "bg");

    // Progress circle (dynamic stroke offset)
    const progress = document.createElementNS(svgNS, "circle");
    progress.setAttribute("cx", "40");
    progress.setAttribute("cy", "40");
    progress.setAttribute("r", radius);
    progress.setAttribute("class", "progress");
    progress.style.strokeDasharray = circumference;
    progress.style.strokeDashoffset = offset;

    // Text in the center of the circle
    const text = document.createElementNS(svgNS, "text");
    text.setAttribute("x", "40");
    text.setAttribute("y", "45");
    text.textContent = `${score}%`;

    // Append elements to SVG
    svg.appendChild(bg);
    svg.appendChild(progress);
    svg.appendChild(text);

    // Wrap SVG in a container div
    const wrapper = document.createElement("div");
    wrapper.className = "circular-progress";
    wrapper.appendChild(svg);

    return wrapper;
}

// ======================================================
// Display recommendation results in the UI
// ======================================================
function displayResults(data) {
    const container = document.getElementById("results");
    container.innerHTML = "";

    // If no recommendations, show a message
    if (!data || (!data.recommended_programs && !data.recommended_unlinked_courses)) {
        container.innerHTML = "<p style='padding:20px;text-align:center;color:#6c757d;'>No recommendations found based on your selected criteria.</p>";
        return;
    }

    // Function to create a single result card
    const createCard = (title, meta, extra, score) => {
        const card = document.createElement("div");
        card.className = "result-card";

        // Clean title from unwanted characters
        const cleanTitle = title.replace(/[[\]"']+/g, '');

        const infoDiv = document.createElement("div");
        infoDiv.className = "card-info";
        infoDiv.innerHTML = `
            <div class="card-title">${cleanTitle}</div>
            <div class="card-meta">${meta}</div>
            <div class="card-extra">${extra}</div>
        `;

        card.appendChild(infoDiv);
        card.appendChild(createCircularProgress(score));

        return card;
    };

    // Render Recommended Programs
    if (data.recommended_programs?.length) {
        const div = document.createElement("div");
        div.innerHTML = "<h3>Recommended Programs</h3>";

        data.recommended_programs.forEach(p => {
            let score = (p.score || 0) * 100;
            score = Math.max(1, Math.min(score, 100));
            score = Math.round(score);

            if (score < 20) return; // Filter out scores below 20%

            const meta = `Type: ${p.degree_type || "N/A"}`;
            const extra = `University: ${p.university || "—"} | Country: ${p.country || "—"} | Language: ${p.language || "—"}`;
            div.appendChild(createCard(p.degree_name, meta, extra, score));
        });

        container.appendChild(div);
    }

    // Render Independent Courses
    if (data.recommended_unlinked_courses?.length) {
        const div = document.createElement("div");
        div.innerHTML = "<h3>📘 Independent Courses</h3>";

        data.recommended_unlinked_courses.forEach(c => {
            let score = (c.score || 0) * 100;
            score = Math.max(1, Math.min(score, 100));
            score = Math.round(score);

            if (score < 20) return; // Filter out scores below 20%

            const meta = `Provider: ${c.provider || "N/A"}`;
            const extra = `University: ${c.university || "—"}`;
            div.appendChild(createCard(c.lesson_name, meta, extra, score));
        });

        container.appendChild(div);
    }
}

// ======================================================
// Perform personalized search based on selected skills and filters
// ======================================================
async function performSearch() {
    const selectedSkills = Array.from(document.querySelectorAll('input[type="checkbox"]:checked'))
        .map(cb => cb.dataset.skillName);

    const payload = {
        target_skills: selectedSkills,
        degree_type: document.getElementById("degreeType").value || null,
        country: document.getElementById("country").value || null,
        language: document.getElementById("language").value || null
    };

    try {
        const data = await fetch(`${API_BASE}/recommend/personalized`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        }).then(r => r.json());

        displayResults(data);

        const recommendedFiltered = (data.recommended_programs || []).filter(p => ((p.score || 0) * 100) >= 20);

        // Εμφάνιση skill graph container μόνο αν υπάρχουν αποτελέσματα
        const skillGraphContainer = document.getElementById("skillGraphContainer");
        if (recommendedFiltered.length > 0) {
            skillGraphContainer.style.display = "block";
            displaySkillGraph(data);
        } else {
            skillGraphContainer.style.display = "none";
        }

    } catch (err) {
        console.error("Error fetching personalized recommendations:", err);
        alert("⚠️ Error fetching results");
    }
}


// ======================================================
// Initialize page on DOMContentLoaded
// ======================================================
document.addEventListener("DOMContentLoaded", () => {
    loadFilters();  // Load dropdown filter options
    loadSkills();   // Load skill categories
    document.getElementById("searchBtn").addEventListener("click", performSearch); // Bind search button
});



// ======================================================
// D3.js Skill Map Visualization - Enhanced
// ======================================================
async function displaySkillGraph(data) {
    // Καθαρισμός προηγούμενου SVG
    d3.select("#skillGraph svg").remove();
    d3.select("#skillGraph div").remove(); // tooltip αν υπάρχει


    const recommended = (data.recommended_programs || []).filter(p => ((p.score || 0) * 100) >= 20);
    if (!recommended.length) return;

    const width = document.getElementById("skillGraph").clientWidth;
    const height = document.getElementById("skillGraph").clientHeight;

    const nodes = [];
    const links = [];
    const skillSet = new Set();

    // Δημιουργία nodes για προγράμματα και links προς δεξιότητες
    recommended.forEach(p => {
        nodes.push({ id: p.degree_name, type: "program", program: p });
        (p.skills || []).forEach(skill => {
            skillSet.add(skill);
            links.push({ source: p.degree_name, target: skill });
        });
    });

    // Δημιουργία nodes για δεξιότητες
    skillSet.forEach(skill => nodes.push({ id: skill, type: "skill" }));

    const svg = d3.select("#skillGraph")
        .append("svg")
        .attr("width", width)
        .attr("height", height);

    const color = d => d.type === "program" ? "#4a90e2" : "#27ae60";

    const simulation = d3.forceSimulation(nodes)
        .force("link", d3.forceLink(links).id(d => d.id).distance(140))
        .force("charge", d3.forceManyBody().strength(-450))
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force("collision", d3.forceCollide(40));

    // Links
    const link = svg.append("g")
        .attr("stroke", "#aaa")
        .attr("stroke-width", 2)
        .selectAll("line")
        .data(links)
        .enter()
        .append("line");

    // Nodes
    const node = svg.append("g")
        .selectAll("circle")
        .data(nodes)
        .enter()
        .append("circle")
        .attr("r", d => d.type === "program" ? 30 : 18)
        .attr("fill", color)
        .call(d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended)
        );

    // Labels
    const label = svg.append("g")
        .selectAll("text")
        .data(nodes)
        .enter()
        .append("text")
        .text(d => d.id)
        .attr("font-size", "12px")
        .attr("text-anchor", "middle")
        .attr("dy", d => d.type === "program" ? 45 : 30);

    // Tooltip
    const tooltip = d3.select("#skillGraph")
        .append("div")
        .style("position", "absolute")
        .style("padding", "6px 10px")
        .style("background", "#fff")
        .style("border", "1px solid #ccc")
        .style("border-radius", "4px")
        .style("pointer-events", "none")
        .style("opacity", 0);

    node.on("mouseover", (event, d) => {
        let html = "";
        if (d.type === "program" && d.program) {
            html = `<b>${d.program.degree_name}</b><br>
                ${d.program.university || ""}<br>
                ${d.program.degree_type || ""}<br>
                Skills: ${(d.program.skills || []).join(", ")}`;
        } else {
            html = `<b>${d.id}</b>`;
        }
        tooltip.html(html)
            .style("left", (event.pageX + 10) + "px")
            .style("top", (event.pageY + 10) + "px")
            .transition().duration(200)
            .style("opacity", 0.95);
    }).on("mouseout", () => {
        tooltip.transition().duration(200).style("opacity", 0);
    });

    // Simulation tick
    simulation.on("tick", () => {
        link
            .attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y);

        node
            .attr("cx", d => d.x)
            .attr("cy", d => d.y);

        label
            .attr("x", d => d.x)
            .attr("y", d => d.y);
    });

    // Drag helpers
    function dragstarted(event, d) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }
    function dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }
    function dragended(event, d) {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }

    const zoom = d3.zoom()
        .scaleExtent([0.5, 3])
        .on("zoom", (event) => {
            svg.selectAll("g, line, circle, text").attr("transform", event.transform);
        });

    svg.call(zoom);


}

