const Test = [
    60.00,
    70.27,
    49.19,
    56.48,
    63.65,
    59.22,
    59.9,
    71.32,
    57.41,
    53.48,
    61.3,
    64.7,
    59.02,
    59.66,
    57.68,
    70.54,
    64.04,
    56.5,
    59.72,
    67.59,
    68.43,
    58.71,
    53.85,
    57.18,
    62.79,
    63.13,
    62.9,
    56.83,
    54.26,
    63.41,
    60.67,
    58.23,
    60.07,
    59.29,
    60.71,
    59.82,
    55.6,
    54.31,
    59.48,
    63.35,
    65.17,
    57.13,
    59.97,
    63.0,
    62.62,
    60.21,
    63.12,
    59.74,
    69.7,
    65.78,
    62.06,
    62.82,
    60.22,
    61.73,
    59.56,
    63.6,
    59.78,
    61.05,
    58.58,
    63.82,
];


const rawData = [
{Time: 1, Accuracy: 74},
{Time: 2, Accuracy: 24},
{Time: 3, Accuracy: 48},
{Time: 4, Accuracy: 44},
{Time: 5, Accuracy: 75},
{Time: 6, Accuracy: 81},
{Time: 7, Accuracy: 4},
{Time: 8, Accuracy: 83},
{Time: 9, Accuracy: 94},
{Time: 10, Accuracy: 77},
{Time: 11, Accuracy: 49},
{Time: 12, Accuracy: 91},
{Time: 13, Accuracy: 55},
{Time: 14, Accuracy: 45},
{Time: 15, Accuracy: 84},
{Time: 16, Accuracy: 2},
{Time: 17, Accuracy: 5},
{Time: 18, Accuracy: 32},
{Time: 19, Accuracy: 83},
{Time: 20, Accuracy: 33},
{Time: 21, Accuracy: 33},
{Time: 22, Accuracy: 29},
{Time: 23, Accuracy: 12},
{Time: 24, Accuracy: 78},
{Time: 25, Accuracy: 86},
{Time: 26, Accuracy: 81},
{Time: 27, Accuracy: 95},
{Time: 28, Accuracy: 76},
{Time: 29, Accuracy: 16},
{Time: 30, Accuracy: 29},
{Time: 31, Accuracy: 64},
{Time: 32, Accuracy: 39},
{Time: 33, Accuracy: 59},
{Time: 34, Accuracy: 39},
{Time: 35, Accuracy: 58},
{Time: 36, Accuracy: 10},
{Time: 37, Accuracy: 78},
{Time: 38, Accuracy: 39},
{Time: 39, Accuracy: 28},
{Time: 40, Accuracy: 46},
{Time: 41, Accuracy: 6},
{Time: 42, Accuracy: 83},
{Time: 43, Accuracy: 16},
{Time: 44, Accuracy: 7},
{Time: 45, Accuracy: 20},
{Time: 46, Accuracy: 79},
{Time: 47, Accuracy: 1},
{Time: 48, Accuracy: 45},
{Time: 49, Accuracy: 98},
{Time: 50, Accuracy: 74},
{Time: 51, Accuracy: 73},
{Time: 52, Accuracy: 21},
{Time: 53, Accuracy: 42},
{Time: 54, Accuracy: 24},
{Time: 55, Accuracy: 4},
{Time: 56, Accuracy: 41},
{Time: 57, Accuracy: 21},
{Time: 58, Accuracy: 72},
{Time: 59, Accuracy: 64},
{Time: 60, Accuracy: 93},
];

var arrayData = [];
const fetchData = async () => {
    try {
        const response = await fetch('/get_npy_data');
        const json = await response.json();
        arrayData = json;
        console.log(arrayData[0][0][0]);
    } catch (error) {
        console.error('Error fetching data:', error);
    }
};

fetchData();

var i = 0;
for (const value of Test) {
        rawData[i].Time=i;
    rawData[i].Accuracy=value;
    i+=1;
}

var xValue,yValue;

const data = rawData.map(o => ({ x: o.Time, y: o.Accuracy}));
const colors = ['red', 'green']; 

const myChart = new Chart(document.getElementById("myChart"), {
    type: "scatter",
    plugins: [{
    afterDraw: chart => {      
        var ctx = chart.chart.ctx; 
        var xAxis = chart.scales['x-axis-1'];
        var yAxis = chart.scales['y-axis-1'];      
        chart.config.data.datasets[0].data.forEach((value, index) => {
        if (index > 0) {        
            var valueFrom = data[index - 1];
            var xFrom = xAxis.getPixelForValue(valueFrom.x);     
            var yFrom = yAxis.getPixelForValue(valueFrom.y); 
            
            var direction = rawData[index -1].direction;
            var Accuracy = rawData[index -1].Accuracy;
            
            var xTo = xAxis.getPixelForValue(value.x);         
            var yTo = yAxis.getPixelForValue(value.y); 
            ctx.save();          
            ctx.strokeStyle = colors[Math.floor(Accuracy / 60)];           
            ctx.lineWidth = 1;
            //ctx.tension=0.4;
            ctx.beginPath();
            ctx.moveTo(xFrom, yFrom);             
            ctx.lineTo(xTo, yTo);
            ctx.stroke();
            ctx.restore();
        }
        });      
    }
    }],
    data: {
    datasets: [{
        label: "Accuracy Score",
        data: data,
        /* borderColor: "rgb(75,192,192)" */
    }]
    },
    options: {
        maintainAspectRatio: false,
    legend: {
        position: "bottom"
        //display: false
    },
onClick: () => {
        console.log('Hovered over point with x value:', yValue.x, 'and y value:', yValue.y);
    },
    tooltips: {
        //enabled: false, // Disable native tooltips
        custom: function(tooltipModel) {
        if (tooltipModel.opacity === 0) {
            return;
        }

        var datasetIndex = tooltipModel.dataPoints[0].datasetIndex;
        var index = tooltipModel.dataPoints[0].index;
        var dataset = myChart.data.datasets[datasetIndex];
        xValue = myChart.data.labels[index];
        yValue = dataset.data[index];

        //console.log('Hovered over point with x value:', xValue, 'and y value:', yValue);
        //console.log('Coordinates (x, y):', tooltipModel.caretX, tooltipModel.caretY);
        }
    },
    scales: {
        yAxes: [{
        ticks: {
            maxTicksLimit: 10
        }
        }]
    },
    }  
});