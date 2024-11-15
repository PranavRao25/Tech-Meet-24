// Variable to keep track of the highlighted node
let highlightedNode = null;

// Function to handle node highlighting (to be changed)
function highlightNode(node) {
    if (highlightedNode) {
        // Remove highlight from previously highlighted node
        highlightedNode.classed("highlighted", false);
    }
    // Highlight the clicked node
    node.classed("highlighted", true);
    highlightedNode = node;  // Update the highlighted node
}

// Function to draw a pipeline with a given vertical offset
function drawPipeline(pipelineData, offsetY) {
    // Create a group for the pipeline
    const group = svg.append("g").attr("transform", `translate(0, ${offsetY})`);

    // Draw links between nodes
    group.selectAll("line")
        .data(pipelineData.slice(1))  // Connect each module to the previous one
        .enter()
        .append("line")
        .attr("x1", (d, i) => pipelineData[i].x + 20)
        .attr("y1", 100)  // Fixed y-position for all nodes within the pipeline
        .attr("x2", d => d.x - 20)
        .attr("y2", 100)
        .attr("stroke", "black");

    // Draw nodes
    group.selectAll("rect")
        .data(pipelineData)
        .enter()
        .append("rect")
        .attr("x", d => d.x - 25)
        .attr("y", 75)  // Fixed y-position for all nodes
        .attr("width", 50)
        .attr("height", 50)
        .attr("fill", "#ddd")
        .attr("class", "node")
        .attr("id", d => d.id.replace(/\s+/g, ''))
        .on("click", function(event, d) {
            highlightNode(d3.select(this));
        });

    // Add labels to nodes
    group.selectAll("text")
        .data(pipelineData)
        .enter()
        .append("text")
        .attr("x", d => d.x)
        .attr("y", 105)
        .attr("text-anchor", "middle")
        .text(d => d.id);
}

// Data for two pipelines without y coordinates for nodes

const queryNode = { id: "Query", x: 200, y: 200 };

const startNode = { id: "MoE", x: 300, y: 200 };

const pipeline1 = [
    { id: "CoT", x: 600 },
    { id: "Reranker", x: 700 }
];

const pipeline2 = [
    { id: "Step Back", x: 500 },
    { id: "MCoT", x: 600 },
    { id: "Reranker", x: 700 }
];

const pipeline3 = [
    { id: "Step Back", x: 500 },
    { id: "ToT", x: 600 },
    { id: "Reranker", x: 700 }
];

const endNode = { id: "Thresholder", x: 900, y: 200 };

const pipeline4 = [
    { id: "LLM", x: 1000 },
];

const pipeline5 = [
    { id: "Web", x: 1000 },
];

const answerNode = { id: "Answer", x: 1200, y: 200 };

// SVG setup
const width = 3000, height = 3000;
const svg = d3.select("#pipelines").append("svg")
    .attr("width", width)
    .attr("height", height);

// Draw shared starting node
const queryNodeElement = svg.append("circle")
    .attr("cx", queryNode.x)
    .attr("cy", queryNode.y)
    .attr("r", 20)
    .attr("fill", "#FFA500")
    .attr("class", "node")
    .attr("id", "Start")
    .on("click", function() {
        highlightNode(d3.select(this));
    });
svg.append("text")
    .attr("x", queryNode.x)
    .attr("y", queryNode.y + 5)
    .attr("text-anchor", "middle")
    .text(queryNode.id);

// Draw shared starting node
const startNodeElement = svg.append("rect")
    .attr("x", startNode.x - 25)
    .attr("y", startNode.y - 20)
    .attr("width", 50)
    .attr("height", 50)
    .attr("fill", "#ddd")
    .attr("class", "node")
    .attr("id", "Start")
    .on("click", function() {
        highlightNode(d3.select(this));
    });
svg.append("text")
    .attr("x", startNode.x)
    .attr("y", startNode.y + 5)
    .attr("text-anchor", "middle")
    .text(startNode.id);

// Draw shared end node
const endNodeElement = svg.append("rect")
    .attr("x", endNode.x - 25)
    .attr("y", endNode.y - 20)
    .attr("width", 50)
    .attr("height", 50)
    .attr("fill", "#ddd")
    .attr("class", "node")
    .attr("id", "End")
    .on("click", function() {
        highlightNode(d3.select(this));
    });
svg.append("text")
    .attr("x", endNode.x)
    .attr("y", endNode.y + 5)
    .attr("text-anchor", "middle")
    .text(endNode.id);

const answerNodeElement = svg.append("circle")
    .attr("cx", answerNode.x)
    .attr("cy", answerNode.y)
    .attr("r", 20)
    .attr("fill", "#FFA500")
    .attr("class", "node")
    .attr("id", "End")
    .on("click", function() {
        highlightNode(d3.select(this));
    });
svg.append("text")
    .attr("x", answerNode.x)
    .attr("y", answerNode.y + 5)
    .attr("text-anchor", "middle")
    .text(answerNode.id);

// Draw lines from shared start node to the first nodes of each pipeline
svg.append("line")
    .attr("x1", queryNode.x + 20)
    .attr("y1", queryNode.y)
    .attr("x2", startNode.x - 20)
    .attr("y2", startNode.y)  // Position of the first node in pipeline1
    .attr("stroke", "black")
    .attr("stroke-width", 2);

svg.append("line")
    .attr("x1", startNode.x + 20)
    .attr("y1", startNode.y)
    .attr("x2", pipeline1[0].x - 20)
    .attr("y2", 100)  // Position of the first node in pipeline1
    .attr("stroke", "black")
    .attr("stroke-width", 2);

svg.append("line")
    .attr("x1", startNode.x + 20)
    .attr("y1", startNode.y)
    .attr("x2", pipeline2[0].x - 20)
    .attr("y2", 200)  // Position of the first node in pipeline2
    .attr("stroke", "black")
    .attr("stroke-width", 2);

svg.append("line")
    .attr("x1", startNode.x + 20)
    .attr("y1", startNode.y)
    .attr("x2", pipeline3[0].x - 20)
    .attr("y2", 300)  // Position of the first node in pipeline3
    .attr("stroke", "black")
    .attr("stroke-width", 2);

// Draw the first pipeline at offsetY = 0 (y = 100)
drawPipeline(pipeline1, 0);

// Draw the second pipeline at offsetY = 200 (y = 300)
drawPipeline(pipeline2, 100);

drawPipeline(pipeline3, 200);

drawPipeline(pipeline4, 50);

drawPipeline(pipeline5, 150);

// Draw lines from last nodes of each pipeline to the shared end node
svg.append("line")
    .attr("x1", pipeline1[pipeline1.length - 1].x + 20)
    .attr("y1", 100)  // Position of the last node in pipeline1
    .attr("x2", endNode.x - 20)
    .attr("y2", endNode.y)
    .attr("stroke", "black")
    .attr("stroke-width", 2);

svg.append("line")
    .attr("x1", pipeline2[pipeline2.length - 1].x + 20)
    .attr("y1", 200)  // Position of the last node in pipeline2
    .attr("x2", endNode.x - 20)
    .attr("y2", endNode.y)
    .attr("stroke", "black")
    .attr("stroke-width", 2);

svg.append("line")
    .attr("x1", pipeline3[pipeline3.length - 1].x + 20)
    .attr("y1", 300)  // Position of the last node in pipeline3
    .attr("x2", endNode.x - 20)
    .attr("y2", endNode.y)
    .attr("stroke", "black")
    .attr("stroke-width", 2);

// Draw line from endNode to next pipelines
svg.append("line")
    .attr("x1", endNode.x + 20)
    .attr("y1", endNode.y)
    .attr("x2", pipeline4[0].x - 20)
    .attr("y2", 150)  // Position of the first node in pipeline4
    .attr("stroke", "black")
    .attr("stroke-width", 2);

svg.append("line")
    .attr("x1", endNode.x + 20)
    .attr("y1", endNode.y)
    .attr("x2", pipeline5[0].x - 20)
    .attr("y2", 250)  // Position of the first node in pipeline5
    .attr("stroke", "black")
    .attr("stroke-width", 2);

svg.append("line")
    .attr("x1", pipeline4[pipeline4.length - 1].x + 20)
    .attr("y1", 150)  // Position of the last node in pipeline2
    .attr("x2", answerNode.x - 20)
    .attr("y2", answerNode.y)
    .attr("stroke", "black")
    .attr("stroke-width", 2);

svg.append("line")
    .attr("x1", pipeline4[pipeline4.length - 1].x + 20)
    .attr("y1", 250)  // Position of the last node in pipeline3
    .attr("x2", answerNode.x - 20)
    .attr("y2", answerNode.y)
    .attr("stroke", "black")
    .attr("stroke-width", 2);