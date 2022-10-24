chart1 = echarts.init(document.getElementById('chart1'))
timeData = []
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
function newtime(nexttime){
	num = count(timeData);
	if(num>300){
		timeData.shift();
		timeData.push(nexttime);
	}
	else{
		timeData.push(nexttime);
	}
}

var actual = [];
var forecast = [];
var absvalue = [];
var latlonData = []
var future_val
function showmeteo(){
	document.getElementById("tem").innerText = MeteoData.tem.toString().substring(0, 5) + '℃';
	document.getElementById("hum").innerText = MeteoData.hum.toString().substring(0, 5) + "%";
	document.getElementById("wind").innerText = MeteoData.wind.toString().substring(0, 5) + 'm/s';
	document.getElementById("press").innerText = MeteoData.press.toString().substring(0, 5) + 'hPa';
        document.getElementById("location").innerText = MeteoData.lat_lon;
    num = count(Evap);
    for(var i=0;i<num;i++){
        timeData.push(Evap[i].times)
        latlonData.push(Evap[i].lat_lon)
	actual.push(Evap[i].evap)
        if(i<num-1){
                forecast.push(Evap[i].futu)
        }
        
        future_val = Evap[i].futu
        if(i>0){
            absvalue.push(Math.abs(actual[i] - forecast[i-1]))
        }
    }
    actual.shift()
    timeData.shift()
}


function nextvalue(newdata){
	//alert('nextvalue')
	num = count(timeData)
    var send_evap = newdata.evap;
    var send_abs = Math.abs(send_evap-future_val)
    var send_latlon = newdata.lat_lon;
	//alert(num)
	document.getElementById("tem").innerText = newdata.tem.toString().substring(0, 5) + '℃';
    document.getElementById("hum").innerText = newdata.hum.toString().substring(0, 5) + "%";
	document.getElementById("wind").innerText = newdata.wind.toString().substring(0, 5) + 'm/s';
	document.getElementById("press").innerText = newdata.press.toString().substring(0, 5) + 'hPa';
	document.getElementById("location").innerText = newdata.lat_lon;
	if(num>299){
		actual.shift()
		forecast.shift()
		absvalue.shift()
                latlonData.shift()
		actual.push(send_evap)
		forecast.push(future_val)
		absvalue.push(send_abs)
                latlonData.push(send_latlon)
	}
	else{
		actual.push(send_evap)
		forecast.push(future_val)
		absvalue.push(send_abs)
                latlonData.push(send_latlon)
	}
	future_val = newdata.futu;
}

option = {
    tooltip: {
        trigger: 'axis',
    },
    legend: {
        data: ['实际值', '预测值', '误差']
    },
    axisPointer: {
        link: {xAxisIndex: 'all'}
    },
    dataZoom: [
		{
            show: true,
            realtime: true,
            start: 90,
            end: 100,
            xAxisIndex: [0, 1]
        },
        {
            type: 'inside',
            realtime: true,
            start: 90,
            end: 100,
            xAxisIndex: [0, 1]
        }
    ],
    grid: [{
        left: 50,
        right: 50,
        height: '33%'
    }, {
        left: 50,
        right: 50,
        top: '55%',
        height: '33%'
    }],
    toolbox: {
        show: true,
        feature: {
            dataZoom: {
                yAxisIndex: 'none'
            },
            dataView: {
                readOnly: true, 
                optionToContent: function(opt) {
                    var axisData = opt.xAxis[0].data;
                    var table = '<table style="width:100%; text-align:center"><tbody><tr>' + 
                                '<td>时间</td><td>实际值</td><td>预测值</td><td>经纬度</td>' + 
                                '</tr>';
                    for (var i=0, l=axisData.length; i<l; i++) {
                        table += '<tr>' +
                                 '<td>' + timeData[i] + '</td>' + 
                                 '<td>' + actual[i] + '</td>' + 
                                 '<td>' + forecast[i] + '</td>' +
                                 '<td>' + latlonData[i] + '</td>' + 
                                 '</tr>';
                    }
                    table += '</tbody></table>';
                    return table;
                }
            },
            magicType: {type: ['line', 'bar']},
            restore: {},
            saveAsImage: {}
        }
    },
    xAxis: [
        {
        type: 'category',
        boundaryGap: false,
        axisLine: {onZero: true},
        data: timeData
        }, 
        {
            gridIndex: 1,
            type: 'category',
            boundaryGap: false,
            axisLine: {onZero: true},
            data: timeData,
            position: 'top'
        }
    ],
    yAxis: [
        {
            gridIndex:0,
            type: 'value',
            name: '高度(m)'
        },
        {
            gridIndex: 1,
            name: '误差(m)',
            type: 'value',
            inverse: true
        }
    ],
    series: [
        {
            name: '实际值',
            type: 'line',
            data: actual,
            xAxisIndex: 0,
            yAxisIndex: 0,
            markLine: {
                data: [
                    {type: 'average', name: '平均值'}
                ]
            }
        },
        {
            name: '预测值',
            type: 'line',
            data: forecast,
            xAxisIndex: 0,
            yAxisIndex: 0,
            itemStyle: {
		    	normal:{
		    		 color: '#00BFFF'
		    	}
		    },
            markLine: {
                data: [
                    {type: 'average', name: '平均值'},
                ]
            }
        }
        ,
		{
		    name: '误差',
		    type: 'line',
		    symbolSize: 8,
			xAxisIndex: 1,
            yAxisIndex: 1,
            markPoint: {
                data:[
                    {type: 'max', name: '最大值'},
                    {type: 'min', name: '最小值'}
                ]

            },
            markLine: {
                data: [
                    {type: 'average', name: '平均值'},
                ]
            },
			itemStyle: {
		    	normal:{
		    		 color: '#EE9A00'
		    	}
		    },
		    hoverAnimation: false,
		    data: absvalue
		}
    ]
};
