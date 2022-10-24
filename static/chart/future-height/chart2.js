var chart2 = echarts.init(document.getElementById("main"))
				
var RSME = [];
var MAPE = [];

function getRSME(y, y_hat){
					return Math.sqrt(Math.abs(y*y-y_hat*y_hat))
				}
function getMAPE(y, y_hat){
        // 避免计算得到0
        if (y==0) y = 0.01
	return Math.abs(y - y_hat) / Math.abs(y)
}
function getMAE(y, y_hat){
	return Math.abs(y - y_hat)
}
function getR2_total(){
	num = count(timeData)
	console.log("num: ", num)
	sum = 0
	for(i=0;i<num;i++){
		sum = sum + actual[i]
	}
	mean = sum/num
	up = 0
	down = 0
	for(i=0;i<num;i++){
		up = up + (actual[i] - forecast[i])*(actual[i] - forecast[i])
		down = down + (actual[i] - mean)*(actual[i] - mean)
	}
	console.log('up', up)
	console.log('down', down)
	R2_total = 1 - up/down
	return R2_total
}
function getMAP_total(){
	num = count(timeData)
	MAP_total = 0
	for(i=0;i<num;i++){
		MAP_total = MAP_total + getMAE(actual[i],forecast[i])
	}
	MAP_total = MAP_total/num
	return MAP_total
}
function getMAPE_total(){
	num = count(timeData)
	MAPE_total = 0
	for(i=0;i<num;i++){
		MAPE_total = MAPE_total + Math.abs(actual[i] - forecast[i])/actual[i]
	}
	MAPE_total = MAPE_total/num
	return MAPE_total
}
function getRSME_total(){
	num = count(timeData)
	RSME_total = 0
	for(i=0;i<num;i++){
		RSME_total = RSME_total + (actual[i] - forecast[i]) * (actual[i] - forecast[i])
	}
	RSME_total = Math.sqrt(RSME_total/num)
	return RSME_total
}
function Total(){
	R2_total = getR2_total();
	console.log(R2_total)
	MAPE_total = getMAPE_total();
	MAP_total = getMAP_total();
	RMSE_total = getRSME_total();
	Perf_T = [MAP_total, MAPE_total*100, RMSE_total, R2_total*100];
	option3.series.data[0].value = Perf_T;
	len = actual.length;
	MAE_last = getMAE(actual[len-1], forecast[len-1]);
	MAPE_last = getMAPE(actual[len-1], forecast[len-1]);
	RMSE_last = getRSME(actual[len-1], forecast[len-1]);
	Perf_L = [MAE_last, MAPE_last*100, RMSE_last, R2_total*100];
	document.getElementById("nextvalue").value = future_val.toFixed(4);
	document.getElementById("rmse").value = RMSE_last.toFixed(4);
	document.getElementById("mape").value = (MAPE_last*100).toFixed(4) + "%";
	document.getElementById("mae").value = MAE_last.toFixed(4);
	document.getElementById("r2").value = (R2_total*100).toFixed(4) + "%";
	option3.series.data[1].value = Perf_L;
}
function push_RSME_MAPE(){
	num = count(timeData);
	i = 0;
	while (i < num) {
		i++;
		RSME.push(getRSME(actual[i], forecast[i]));
		MAPE.push(getMAPE(actual[i], forecast[i]))
	}
}
function renew_RSME_MAPE(){
	num = count(timeData);
	if(num > 300){
		RSME.shift();
		MAPE.shift();
		RSME.push(getRSME(actual[num-1], forecast[num-1]));
		MAPE.push(getMAPE(actual[num-1], forecast[num-1]))
	}
	else{
		//alert(getRSME(actual[num-1], forecast[num-1]))
		RSME.push(getRSME(actual[num-1], forecast[num-1]));
		MAPE.push(getMAPE(actual[num-1], forecast[num-1]))
	}
}
option1 = {
	title: {
		text: "蒸发波导高度预测评估",
		left: 'center'
	},
	tooltip: {
		trigger: "axis"
	},
	grid: [
		{
			left: "5%",
			top: "20%",
			width: "40%",
			height: "70%"
		},
		{
			left: "55%",
			top: "20%",
			width: "40%",
			height: "70%"
		}
		],
	xAxis: [
		{
			data: timeData,
		},
		{
			gridIndex: 1,
			data: timeData,
		}
		],
	yAxis: [
		{
			gridIndex:0,
			name: "均方根误差"
		},
		{
			name: "平均绝对百分比误差(%)",
			gridIndex: 1,
		}
		],
	toolbox: {
		left: "right",
		feature: {
			dataZoom: {
				yAxisIndex: "none"
			},
			restore: {},
			saveAsImage: {}
		}
		},
	dataZoom: [
		{
			type: 'inside',
			show: true,
			realtime: true,
			start: 90,
			end: 100,
			xAxisIndex: [0, 1]
		},
		{
						
			show: true,
			realtime: true,
			start: 90,
			end: 100,
			xAxisIndex: [0, 1]
		}
		],
	visualMap: [
		{
			seriesIndex: 0,
			top: "8%",
			right: 600,
			pieces: [
				{
					gt: 0,
					lte: 3,
					color: "#096"
				},
				{
					gt: 3,
					lte: 5,
					color: "#ffde33"
				},
				{
					gt: 5,
					lte: 7,
					color: "#ff9933"
				},
				{
					gt: 7,
					color: "#660099"
				}
				],
			outOfRange: {
				color: "#999"
			}
			},
		{
			seriesIndex: 1,
			top: "8%",
			right: 50,
			pieces: [
				{
					gt: 0,
					lte: 10,
					color: "#096"
				},
				{
					gt: 10,
					lte: 25,
					color: "#ffde33"
				},
				{
					gt: 25,
					lte: 50,
					color: "#ff9933"
				},
				{
					gt: 50,
					color: "#660099"
				}
				],
			outOfRange: {
				color: "#999"
			}
		}
		],
	series: [
		{
			name: "均方根误差",
			type: "bar",
			data: RSME
		},
		{
			name: "平均绝对百分比误差",
			type: "bar",
			xAxisIndex: 1,
			yAxisIndex: 1,
			data: MAPE
		}
		]
}

function setoption(newdata){
	var send_time = newdata.times;
	newtime(send_time);
	nextvalue(newdata);
	document.getElementById("nextvalue1").innerHTML = future_val;
	renew_RSME_MAPE()
	Total()
	chart1.setOption(option)
	chart2.setOption(option1)
	chart3.setOption(option3)
}
showmeteo()
push_RSME_MAPE()
Total()
chart1.setOption(option)
chart2.setOption(option1)
chart3.setOption(option3)
$(document).ready(function() {
    namespace = '/future';
    var socket = io.connect(location.protocol + '//' + document.domain + ':' + location.port + namespace, {transports: ['websocket']});
    socket.on('server_response', function(res) {
        console.log(res.futu);
        setoption(res.futu)
    });
});
				