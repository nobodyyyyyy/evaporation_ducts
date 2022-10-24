var Chart_one = echarts.init(document.getElementById('echarts'));	
var Chart_two = echarts.init(document.getElementById('echarts2'));
var Chart_tri = echarts.init(document.getElementById('echarts3'));
var Chart_for = echarts.init(document.getElementById('echarts4'));

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

var timeData = []
var humData = []
var temData = []
var pressData = []
var windData = []
var dirtData = []
var latlonData = []

function randomvalue(){
    data_num = Data.length;
    for (var i=0;i<data_num;i++) {
        timeData.push(Data[i].times)
        temData.push(Data[i].tem)
        pressData.push(Data[i].press)
        humData.push(Data[i].hum)
        windData.push(Data[i].wind)
        dirtData.push(Data[i].direction)
        latlonData.push(Data[i].lat_lon)
    }
    console.log("init data num: ", data_num)
    document.getElementById("tem").innerHTML = temData[data_num-1].toString().substring(0, 5) + '℃';
    document.getElementById("hum").innerHTML = humData[data_num-1].toString().substring(0, 5) + "%";
    document.getElementById("wind").innerHTML = windData[data_num-1].toString().substring(0, 6) + 'm/s';
    document.getElementById("press").innerHTML = pressData[data_num-1].toString().substring(0, 6) + 'hPa';
    document.getElementById("location").innerHTML = latlonData[data_num-1];
    var TEXT = document.getElementById("datatext");
    content_temp = ""
    content_temp += "<b>" + timeData[data_num-1] + "</b>:<br/>" + "当前采集到的温度为：" + document.getElementById("tem").innerHTML
    content_temp += "<br/>当前采集到的湿度为：" + document.getElementById("hum").innerHTML + "；"
    content_temp += "<br/>当前采集到的风速为：" + document.getElementById("wind").innerHTML + "，风向为：" + dirtData[data_num-1].toString().substring(0,8)
    content_temp += "<br/>当前采集到的湿度为：" + document.getElementById("press").innerHTML + "。"
    content_temp += "<br/>"
	//content_text = content_temp+content_text
    TEXT.innerHTML = content_temp
}

content_text = ""
lastlength = []
function nextvalue(newdata){
	num = temData.length
	var TEM = document.getElementById("tem");
	var HUM = document.getElementById("hum");
	var WIND = document.getElementById("wind");
	var PRESS = document.getElementById("press");
	var TEXT = document.getElementById("datatext");
	var LOCATION = document.getElementById("location")
	content_temp = "";
	if(num>299){
                timeData.shift();
		temData.shift();
                pressData.shift();
                humData.shift();
                windData.shift();
                dirtData.shift();
                latlonData.shift(); 

                latlonData.push(newdata.lat_lon)
                timeData.push(newdata.times)
                temData.push(newdata.tem)
                pressData.push(newdata.press)
                humData.push(newdata.hum)
                windData.push(newdata.wind)
                dirtData.push(newdata.direction)

		TEM.innerHTML = temData[299].toString().substring(0, 5) + '℃';
		HUM.innerHTML = humData[299].toString().substring(0, 5) + "%";
		WIND.innerHTML = windData[299].toString().substring(0, 5) + 'm/s';
		PRESS.innerHTML = pressData[299].toString().substring(0, 5) + 'hPa';
                LOCATION.innerHTML = latlonData[299];
		content_temp += "<b>" + timeData[299] + "</b>:<br/>" + "当前采集到的温度为：" + TEM.innerHTML
		content_temp += "<br/>当前采集到的湿度为：" + HUM.innerHTML + "；"
		content_temp += "<br/>当前采集到的风速为：" + WIND.innerHTML + "，风向为：" + dirtData[299].toString().substring(0,4)
		content_temp += "<br/>当前采集到的湿度为：" + PRESS.innerHTML + "。"
	}
	else{
		timeData.push(newdata.times)
                temData.push(newdata.tem)
                pressData.push(newdata.press)
                humData.push(newdata.hum)
                windData.push(newdata.wind)
                dirtData.push(newdata.direction)
                latlonData.push(newdata.latlon)

		TEM.innerHTML = temData[num].toString().substring(0, 5) + '℃';
		HUM.innerHTML = humData[num].toString().substring(0, 5) + "%";
		WIND.innerHTML = windData[num].toString().substring(0, 5) + 'm/s';
		PRESS.innerHTML = pressData[num].toString().substring(0, 5) + 'hPa';
                LOCATION.innerHTML = latlonData[num];
		content_temp += "<b>" + timeData[num] + "</b>:<br/>" + "当前采集到的温度为：" + TEM.innerHTML
		content_temp += "<br/>当前采集到的湿度为：" + HUM.innerHTML + "；"
		content_temp += "<br/>当前采集到的风速为：" + WIND.innerHTML + "；风向为：" + dirtData[num].toString().substring(0,4)
		content_temp += "<br/>当前采集到的湿度为：" + PRESS.innerHTML + "。"
	}
	content_temp += "<br/>"
	content_text = content_temp + content_text
	if(content_text.length > 1000){
		console.log("true")
		content_text = content_text.substring(0, content_text.length-lastlength[0])
		lastlength.shift()
		TEXT.innerHTML = content_text
	}
	else{
		TEXT.innerHTML = content_text
	}
	lastlength.push(content_temp.length)
	
}

function setoption(newdata){
    nextvalue(newdata);
    Chart_one.setOption(option);
    Chart_two.setOption(option2);
    Chart_tri.setOption(option3);
    Chart_for.setOption(option4);
}

randomvalue()
option = {
    tooltip: {
    trigger: 'axis',
    formatter: function (params) {
      params = params[0]
      return (
        '地理位置：' + latlonData[params.dataIndex] + 
        '<br />时间： ' + timeData[params.dataIndex] + 
        '<br />温度： ' + temData[params.dataIndex]
      );
    },
    axisPointer: {
      animation: false
    }
  },
    legend: {
        data: ['温度'],
        left: 10
    },
    toolbox: {
        feature: {
            dataZoom: {
                yAxisIndex: 'none'
            },
            dataView: {
                readOnly: true,
                optionToContent: function(opt) {
                    var axisData = opt.xAxis[0].data;
                    var table = '<table style="width:100%; text-align:center"><tbody><tr>' + 
                                '<td>时间</td><td>温度</td><td>地理位置</td>' + 
                                '</tr>';
                    for (var i=0, l=axisData.length; i<l; i++) {
                        table += '<tr>' +
                                 '<td>' + timeData[i] + '</td>' + 
                                 '<td>' + temData[i] + '</td>' + 
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
    axisPointer: {
        link: {xAxisIndex: 'all'}
    },
    dataZoom: [
        {
            show: true,
            realtime: true,
            start: 80,
            end: 100,
        }
    ],
    xAxis: [
        {
            type: 'category',
            boundaryGap: false,
            axisLine: {onZero: true},
            data: timeData
        }
    ],
    yAxis: [
        {
            name: '气温(℃)',
            type: 'value',
        }
    ],
    series: [
        {
            name: '温度',
            type: 'line',
            symbolSize: 8,
			itemStyle: {
            	normal:{
            		 color: '#FF8247'
            	}
            },
			markPoint: {
			    data: [
			        {type: 'max', name: '最大值'},
			        {type: 'min', name: '最小值'}
			    ]
			},
			markLine: {
			    data: [
			        {type: 'average', name: '平均值'}
			    ]
			},
            hoverAnimation: false,
            data: temData
        }
    ]
};
option2 = {
    tooltip: {
        trigger: 'axis',
        formatter: function (params) {
            params = params[0]
            return (
                '地理位置：' + latlon[params.dataIndex] + 
                '<br />时间： ' + timeData[params.dataIndex] + 
                '<br />气压： ' + pressData[params.dataIndex]
            );
        },
        axisPointer: {
            animation: false
        }
    },
    legend: {
        data: ['气压'],
        left: 10
    },
    toolbox: {
        feature: {
            dataZoom: {
                yAxisIndex: 'none'
            },
            dataView: {
                readOnly: true, 
                optionToContent: function(opt) {
                    var axisData = opt.xAxis[0].data;
                    var table = '<table style="width:100%; text-align:center"><tbody><tr>' + 
                                '<td>时间</td><td>气压</td><td>地理位置</td>' + 
                                '</tr>';
                    for (var i=0, l=axisData.length; i<l; i++) {
                        table += '<tr>' +
                                 '<td>' + timeData[i] + '</td>' + 
                                 '<td>' + pressData[i] + '</td>' + 
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
    axisPointer: {
        link: {xAxisIndex: 'all'}
    },
    dataZoom: [
        {
            show: true,
            realtime: true,
            start: 80,
            end: 100
        }
    ],
    xAxis: [
        {
            type: 'category',
            boundaryGap: false,
            axisLine: {onZero: true},
            data: timeData
        }
    ],
    yAxis: [
        {
            name: '气压(hPa)',
            type: 'value',
        }
    ],
    series: [
        {
            name: '气压',
            type: 'line',
            symbolSize: 8,
            hoverAnimation: false,
            data: pressData,
			markPoint: {
                data: [
                    {type: 'max', name: '最大值'},
                    {type: 'min', name: '最小值'}
                ]
            },
            markLine: {
                data: [
                    {type: 'average', name: '平均值'}
                ]
            },
        }
    ]
};
option3 = {
    tooltip: {
        trigger: 'axis', 
        formatter: function (params) {
            params = params[0]
            return (
                '地理位置：' + latlonData[params.dataIndex] + 
                '<br />时间： ' + timeData[params.dataIndex] + 
                '<br />湿度： ' + humData[params.dataIndex]
            );
        },
        axisPointer: {
            animation: false
        }
    },
    legend: {
        data: ['湿度'],
        left: 10
    },
    toolbox: {
        feature: {
            dataZoom: {
                yAxisIndex: 'none'
            },
            dataView: {
                readOnly: true, 
                optionToContent: function(opt) {
                    var axisData = opt.xAxis[0].data;
                    var table = '<table style="width:100%; text-align:center"><tbody><tr>' + 
                                '<td>时间</td><td>湿度</td><td>地理位置</td>' + 
                                '</tr>';
                    for (var i=0, l=axisData.length; i<l; i++) {
                        table += '<tr>' +
                                 '<td>' + timeData[i] + '</td>' + 
                                 '<td>' + humData[i] + '</td>' + 
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
    axisPointer: {
        link: {xAxisIndex: 'all'}
    },
    dataZoom: [
        {
            show: true,
            realtime: true,
            start: 80,
            end: 100,
        }
    ],
    xAxis: [
        {
            type: 'category',
            boundaryGap: false,
            axisLine: {onZero: true},
            data: timeData
        }
    ],
    yAxis: [
        {
            name: '湿度(%)',
            type: 'value',
        }
    ],
    series: [
        {
            name: '湿度',
            type: 'line',
            symbolSize: 8,
			itemStyle: {
            	normal:{
            		 color: '#87CEFA'
            	}
            },
			markPoint: {
			    data: [
			        {type: 'max', name: '最大值'},
			        {type: 'min', name: '最小值'}
			    ]
			},
			markLine: {
			    data: [
			        {type: 'average', name: '平均值'}
			    ]
			},
            hoverAnimation: false,
            data: humData
        }
    ]
};
option4 = {
    tooltip: {
        trigger: 'axis',
        formatter: function (params) {
            params = params[0]
            return (
                '地理位置：' + latlonData[params.dataIndex] + 
                '<br />时间： ' + timeData[params.dataIndex] + 
                '<br />风速： ' + windData[params.dataIndex] + 
                '<br/>风向：' + dirtData[params.dataIndex]
            );
        },
        axisPointer: {
            animation: false
        }
    },
    legend: {
        data: ['风速', '风向'],
        left: 10
    },
    toolbox: {
        feature: {
            dataZoom: {
                yAxisIndex: 'none'
            },
            dataView: {
                readOnly: true, 
                optionToContent: function(opt) {
                    var axisData = opt.xAxis[0].data;
                    var table = '<table style="width:100%; text-align:center"><tbody><tr>' + 
                                '<td>时间</td><td>风速</td><td>风向</td><td>地理位置</td>' + 
                                '</tr>';
                    for (var i=0, l=axisData.length; i<l; i++) {
                        table += '<tr>' +
                                 '<td>' + timeData[i] + '</td>' + 
                                 '<td>' + windData[i] + '</td>' + 
                                 '<td>' + dirtData[i] + '</td>' +
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
    axisPointer: {
        link: {xAxisIndex: 'all'}
    },
    dataZoom: [
		{
            show: true,
            realtime: true,
            start: 80,
            end: 100,
            xAxisIndex: [0, 1]
        },
        {
            type: 'inside',
            realtime: true,
            start: 50,
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
            name: '风速(m/s)',
            type: 'value',
        },
        {
            gridIndex: 1,
            name: '风向(°)',
            type: 'value',
            inverse: true
        }
    ],
    series: [
        {
            name: '风速',
            type: 'line',
            symbolSize: 8,
			itemStyle: {
            	normal:{
            		 color: '#7FFFD4'
            	}
            },
			markPoint: {
			    data: [
			        {type: 'max', name: '最大值'},
			        {type: 'min', name: '最小值'}
			    ]
			},
			markLine: {
			    data: [
			        {type: 'average', name: '平均值'}
			    ]
			},
            hoverAnimation: false,
            data: windData
        },
		{
		    name: '风向',
		    type: 'bar',
		    symbolSize: 8,
			xAxisIndex: 1,
            yAxisIndex: 1,
			itemStyle: {
		    	normal:{
		    		 color: '#54FF9F'
		    	}
		    },
		    hoverAnimation: false,
		    data:dirtData
		}
    ]
};

Chart_one.setOption(option);
Chart_two.setOption(option2);
Chart_tri.setOption(option3);
Chart_for.setOption(option4);

		