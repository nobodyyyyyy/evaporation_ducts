var chart_left = echarts.init(document.getElementById('chart-left'))
var timeData = [];

function updateNowTime() {
    var nowDate = new Date();
    var year = nowDate.getFullYear();
    var month = nowDate.getMonth() + 1;
    var day = nowDate.getDate();
    var hour = nowDate.getHours();
    var minute = nowDate.getMinutes();
    var second = nowDate.getSeconds();
    document.getElementById("contentscroll2").innerHTML += "\n已成功连接服务器：" + year.toString() + "-" + 
                                                       month.toString() + "-" + 
                                                       day.toString() + " " +
                                                       hour.toString() + ":" + 
                                                       minute.toString() + ":" +
                                                       second.toString() + "\n";
}
updateNowTime()
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
	if(num>299){
		timeData.shift();
		timeData.push(nexttime);
	}
	else{
		timeData.push(nexttime);
	}

}

var height = []
var latlon = []
function showmeteo()
{
	document.getElementById("tem").innerText = MeteoData.tem.toString().substring(0, 5) + '℃';
	document.getElementById("hum").innerText = MeteoData.hum.toString().substring(0, 5) + "%";
	document.getElementById("wind").innerText = MeteoData.wind.toString().substring(0, 5) + 'm/s';
	document.getElementById("press").innerText = MeteoData.press.toString().substring(0, 5) + 'hPa';
	document.getElementById("location").innerHTML = MeteoData.lat_lon;
	data_num = count(Evap);
	for(i=0;i<data_num;i++){
                timeData.push(Evap[i].times);
		height.push(Evap[i].evap);
                latlon.push(Evap[i].lat_lon);
        }
}
showmeteo()
function nextvalue(newdata){
    document.getElementById("tem").innerText = newdata.tem.toString().substring(0, 5) + '℃';
    document.getElementById("hum").innerText = newdata.hum.toString().substring(0, 5) + "%";
    document.getElementById("wind").innerText = newdata.wind.toString().substring(0, 5) + 'm/s';
    document.getElementById("press").innerText = newdata.press.toString().substring(0, 5) + 'hPa';
    document.getElementById("location").innerHTML = newdata.lat_lon;
    data_num = count(height)
    var send_evap = newdata.evap;
    var send_latlon = newdata.lat_lon;
    if(data_num>299){
        height.shift()
        latlon.shift()
        height.push(send_evap)
        latlon.push(send_latlon)
    }
    else{
        height.push(send_evap)
        latlon.push(send_latlon)
    }
}

option = {
        tooltip: {
            trigger: 'axis',
            formatter: function (params) {
            params = params[0]
            return (
                '地理位置：' + latlon[params.dataIndex] + 
                '<br />时间： ' + timeData[params.dataIndex] + 
                '<br />波导高度： ' + height[params.dataIndex]
            );
            },
        },
        xAxis: {
			type: 'category',
			boundaryGap: false,
			axisLine: {onZero: true},
            data: timeData
        },
        yAxis: {
            splitLine: {
                show: false
            }
        },
        toolbox: {
            left: 'center',
            feature: {
                dataZoom: {
                    yAxisIndex: 'none'
                },
		dataView: {
                    readOnly: true, 
                    optionToContent: function(opt) {
                    var axisData = opt.xAxis[0].data;
                    var table = '<table style="width:100%; text-align:center"><tbody><tr>' + 
                                '<td>时间</td><td>蒸发波导高度</td><td>地理位置</td>' + 
                                '</tr>';
                    for (var i=0, l=axisData.length; i<l; i++) {
                        table += '<tr>' +
                                 '<td>' + timeData[i] + '</td>' + 
                                 '<td>' + height[i] + '</td>' + 
                                 '<td>' + latlon[i] + '</td>' +
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
        dataZoom: [
            {
                show: true,
                realtime: true,
                start: 95,
                end: 100,
            }
        ],
        visualMap: {
            top: 10,
            right: 10,
            pieces: [{
                gt: 0,
                lte: 5,
                color: '#FF83FA'
            }, {
                gt: 5,
                lte: 10,
                color: '#EE7AE9'
            }, {
                gt: 10,
                lte: 15,
                color: '#CD69C9'
            }, {
                gt: 15,
                lte:20,
                color: '#8B4789'
            }],
            outOfRange: {
                color: '#68228B'
            }
        },
        series: {
            name: 'evaporation wave',
            type: 'line',
            data: height,
            markLine: {
                silent: true,
                data: [{
                    yAxis: 5
                }, {
                    yAxis: 10
                }, {
                    yAxis: 15
                }, {
                    yAxis: 20
                }, {
                    yAxis: 25
                }]
            }
        }
    };

chart_left.setOption(option);
function setoption(newdata){
	var send_times = newdata.times;
	newtime(send_times);
	nextvalue(newdata);
	chart_left.setOption(option);
}
