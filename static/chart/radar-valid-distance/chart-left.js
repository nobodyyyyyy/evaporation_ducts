var dom = document.getElementById("chart-left");
var chart_left = echarts.init(dom);
var app = {};
option_left = null;
var colors = [ '#FFB100'];
var symbolSize = 20;
var radar_15_data = [[0, 110], [3, 125], [5, 115], [7, 120], [10, 118],[16,128],[18,121],[40,155],[45,160]];

var updateCount = 0;
var ifUpdate = false;
//
// var press = []
// var tem = []
// var hum = []
// var wind = []
function count(o){
	var t = typeof o;
	if(t == 'string'){
		return o.length;
	}else if(t == 'object'){
		var n = 0;
		for(var i in o){
			n++;
		}
		return n;
	}
	return false;
}
function nextvalue(newdata){
	var TEM = document.getElementById("tem");
	var HUM = document.getElementById("hum");
	var WIND = document.getElementById("wind");
	var PRESS = document.getElementById("press");

	// hum.push(newdata.hum)
	// tem.push(newdata.tem)
	// wind.push(newdata.wind)
	// press.push(newdata.press)
	console.log(1231414214)
	TEM.innerHTML = newdata.hum.toString().substring(0, 5) + '℃'
	HUM.innerHTML = newdata.tem.toString().substring(0, 5) + "%"
	WIND.innerHTML = newdata.wind.toString().substring(0, 5) + 'm/s'
	PRESS.innerHTML = newdata.press.toString().substring(0, 5) + 'hPa'

	if(updateCount == 5){
		radar_15_data[0] = [Math.floor(Math.random()*10-5+5),Math.floor(Math.random()*20-10+110)];
		radar_15_data[1] = [Math.floor(Math.random()*10-5+15),Math.floor(Math.random()*20-10+125)];
		radar_15_data[2] = [Math.floor(Math.random()*10-5+25),Math.floor(Math.random()*20-10+125)];
		radar_15_data[3] = [Math.floor(Math.random()*10-5+35),Math.floor(Math.random()*20-10+125)];
		radar_15_data[4] = [Math.floor(Math.random()*10-5+45),Math.floor(Math.random()*20-10+140)];
		radar_15_data[5] = [Math.floor(Math.random()*10-5+55),Math.floor(Math.random()*20-10+140)];
		radar_15_data[6] = [Math.floor(Math.random()*10-5+65),Math.floor(Math.random()*20-10+140)];
		radar_15_data[7] = [Math.floor(Math.random()*10-5+75),Math.floor(Math.random()*20-10+140)];
		radar_15_data[8] = [Math.floor(Math.random()*10-5+150),Math.floor(Math.random()*20-10+150)];
		radar_15_data[9] = [Math.floor(Math.random()*10-5+180),Math.floor(Math.random()*20-10+150)];
	}
}

option_left = {
  color: colors,
  toolbox: {
	show: true,
	right:'50%',
	feature: {
		dataZoom: {
			yAxisIndex: 'none'
		},
		dataView: {readOnly: true},
		restore: {},
		saveAsImage: {}
	}
  },
  tooltip: {
	   trigger: 'axis',
	   axisPointer: {
		   animation: false
	   }
   },
   legend: {
	  data: ['雷达探测距离'],
	  left: 10,
  },
  grid: {
	   left: '5%',
	   
	   width:'75%'
   },
  xAxis: {
	  name:'(距离：km )',
	  min: 0,
	  max: 200,
	  type: 'value',
	  minInterval:50,
	  axisLine: {onZero: false}
  },
  yAxis: {
	  
	  min: 100,
	  max: 160,
	  type: 'value',
	  inverse:true,
	  minInterval:20,
	  axisLine: {onZero: false}
  },
  dataZoom: [
	  {
		  type: 'slider',
		  yAxisIndex: 0,
		  calculable: true,
	  },
	  {
		  type: 'inside',
		  xAxisIndex: 0,
		  filterMode: 'empty'
	  },
	  {
		  type: 'inside',
		  yAxisIndex: 0,
		  filterMode: 'empty'
	  }
  ],
  series: [
	  {
		  name: '雷达探测距离',
		  id: 'a',
		  type: 'line',
		  smooth: true,
		  data: radar_15_data,
	  }
  ]
};


setTimeout(function () {
  // Add shadow circles (which is not visible) to enable drag.
  chart_left.setOption({
	  graphic: echarts.util.map(radar_15_data, function (item, dataIndex) {
		  return {
			  type: 'circle',
			  position: chart_left.convertToPixel('grid', item),
			  shape: {
				  cx: 0,
				  cy: 0,
				  r: symbolSize / 2
			  },
			  invisible: true,
			  draggable: true,
			  ondrag: echarts.util.curry(onPointDragging, dataIndex),
			  onmousemove: echarts.util.curry(showTooltip, dataIndex),
			  onmouseout: echarts.util.curry(hideTooltip, dataIndex),
			  z: 100
		  };
	  })
  });
}, 0);

window.addEventListener('resize', updatePosition);

chart_left.on('dataZoom', updatePosition);

function updatePosition() {
	chart_left.setOption({
	  graphic: echarts.util.map(radar_15_data, function (item, dataIndex) {
		  return {
			  position: chart_left.convertToPixel('grid', item)
		  };
	  })
  });
}

function showTooltip(dataIndex) {
	chart_left.dispatchAction({
	  type: 'showTip',
	  seriesIndex: 0,
	  dataIndex: dataIndex
  });
}

function hideTooltip(dataIndex) {
	chart_left.dispatchAction({
	  type: 'hideTip'
  });
}

function onPointDragging(dataIndex, dx, dy) {
  data[dataIndex] = chart_left.convertFromPixel('grid', this.position);
  // Update data
  chart_left.setOption({
	  series: [{
		  id: 'a',
		  data: radar_15_data
	  }]
  });
};


if (option_left && typeof option_left === "object") {
	chart_left.setOption(option_left, true);
}