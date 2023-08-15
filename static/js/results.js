const Frames= [1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    ];
    
    const Values = [
    45.56171542164036,
    53.73341267094989,
    56.518501987459835,
    66.09877670578594,
    66.55321174341653,
    74.43590154478291,
    73.14923926189608,
    75.1896231452401,
    80.9061975884643,
    78.09160767161063,
    66.04567950274763,
    70.37170278641432,
    60.98766415927636,
    57.920961593793436,
    51.31931563869888,
    45.87306597442544,
    42.98812819210686,
    47.755213117613444,
    53.30095109439455,
    58.99124828230843,
    58.848209430731984,
    66.74589185383306,
    70.62871945269784,
    75.03751044235406,
    74.0361809363608,
    64.9074911981819,
    62.87631090509935,
    63.93757573425596,
    61.88748551012231,
    56.30447241633871,
    47.29275860349437,
    40.13687386918421,
    ];
    
    
     
    /*  
     var mean=75.0,mean2=70.0;
     var dev=10.0;
     //console.log("[");
     
     var k=0;
     for(var i=0; i<=15;i++){
      //double sign = Math.random()>0.5?1.0:-1.0;
      var sign=-1.0;
      var diff= (i>7.5)?i-7.5:7.5-i;
      //var x = mean - 4*Math.abs(i-7.5) + Math.random()*dev ;
      var x = mean - Math.abs(i-7.5)*Math.abs(i-7.5) + Math.random()*dev ;
       
      Values[k++]=x;
      //console.log(x+",");
     }
     for(var i=0; i<=15;i++){
      //double sign = Math.random()>0.5?1.0:-1.0;
      var sign=-1.0;
      var diff= (i>7.5)?i-7.5:7.5-i;
      var x = mean2 - 4*Math.abs(i-7.5) + Math.random()*dev ;
     
       Values[k++]=x;
      //console.log(x+",");
     }
     //console.log("]");
     */
            
    
    var colors = []
    for(var i=0 ; i<30; i++){
        if(Values[i]>60)
          colors[i]='green';
       else
           colors[i]='red';
    }
    
    
    const ctx = document.getElementById('chart_id').getContext('2d');
    const myChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: Frames,
            datasets: [{
                label: 'Accuracy Score',
                data: Values,
                borderRadius: 15,
                backgroundColor: colors,
                //barThickness:5,
                barPercentage: 0.5
            }]
        },
        options: {
          plugins: {
            legend: {
              //position: "bottom"
              display: false
            },
          },  
          scales: {
            x: {
              display: true,
              title: {
                display: true,
                text: 'Accuracy Score'
              },
              grid: {
                display: false
              }
            },
            y: {
              grid: {
                display: false
              }
            }
          }, 
          maintainAspectRatio: false,
          onClick: function(event, clickedElements){
            if (clickedElements.length === 0) return
    
            const { dataIndex, raw } = clickedElements[0].element.$context
            const barLabel = event.chart.data.labels[dataIndex]
    
            console.log(raw); 			//Accuracy Score
            console.log(barLabel);	//Frame No;
          }
        }
    });