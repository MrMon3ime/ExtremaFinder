document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById("extrema-form");
    const functionInput = document.getElementById("function");
    const resultsSection = document.getElementById("results");
    const derivativeResult = document.getElementById("derivativeResult");
    const criticalPointsResult = document.getElementById("criticalPointsResult");
    const hessianMatrixResult = document.getElementById("hessianMatrixResult");
    const globalPointsResult = document.getElementById("globalPointsResult");
    const plotContainer = document.getElementById("plot");

    form.addEventListener("submit", async function (event) {
        event.preventDefault();

        let functionStr = functionInput.value.trim();
        if (!functionStr) {
            alert("Please enter a valid function!");
            return;
        }

        // Show loading animation
        resultsSection.style.opacity = "0.5";
        resultsSection.innerHTML += `<p class="loading">Analyzing function...</p>`;

        try {
            let response = await fetch("/find_extrema", {
                method: "POST",
                headers: { "Content-Type": "application/x-www-form-urlencoded" },
                body: new URLSearchParams({ function: functionStr })
            });

            let data = await response.json();
            resultsSection.style.opacity = "1"; 
            document.querySelector(".loading").remove();

            if (data.error) {
                resultsSection.innerHTML += `<p class="error">Error: ${data.error}</p>`;
                return;
            }

            // Smoothly update results with animations
            derivativeResult.innerHTML = `<span class="fade-in">${data.gradients.join(", ")}</span>`;
            criticalPointsResult.innerHTML = `<span class="fade-in">${JSON.stringify(data.critical_points)}</span>`;
            hessianMatrixResult.innerHTML = `<span class="fade-in">${data.hessian}</span>`;
            globalPointsResult.innerHTML = `<span class="fade-in">${JSON.stringify(data.classifications)}</span>`;

            // Generate the 3D Plot
            generatePlot(functionStr);

        } catch (error) {
            resultsSection.innerHTML += `<p class="error">Failed to fetch results. Try again!</p>`;
        }
    });

    function generatePlot(funcStr) {
        // Plotly visualization (Placeholder for now)
        plotContainer.innerHTML = "";
        let trace = {
            z: [[0, 1], [1, 0]], 
            type: "surface"
        };
        let layout = {
            title: `Graph of ${funcStr}`,
            autosize: true
        };
        Plotly.newPlot("plot", [trace], layout);
    }
});
