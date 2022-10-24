var chart_left = echarts.init(document.getElementById("chart-left"))
// var app = {};
// 雷达的增益，初始化为一个200*200全0的值
var radar_data = [];
var elec_min = 0;
var elec_max = 200;

function initial_radar() {
    if (elec_data == null) return;
    elec_min = elec_data[0][2];
    elec_max = elec_data[0][2];
	if(Lossflag>0){
		for(i=0;i<200;i++){
			for(j=0;j<200;j++){
			    if (elec_min > elec_data[i*200+j][2]) elec_min = elec_data[i*200+j][2];
			    if (elec_max < elec_data[i*200+j][2]) elec_max = elec_data[i*200+j][2];
			    radar_data.push([elec_data[i*200+j][0], elec_data[i*200+j][1], elec_data[i*200+j][2]]);
				if(elec_data[i*200+j][2]>Lossflag) radar_data[i*200+j][2] = 1;
				else radar_data[i*200+j][2] = 0;
			}
		}
	}
	else{
		for(i=0;i<200;i++){
			for(j=0;j<200;j++){
			    radar_data.push([elec_data[i*200+j][0], elec_data[i*200+j][1], elec_data[i*200+j][2]]);
				radar_data[i*200+j][2] = 0;
			}
		}
	}
}

initial_radar();
option_left = {
    tooltip: {
        formatter: function (params) {
            let res = function () {
                if (params.data[2]) return "有";
                return "无";
            };
            return (
                '高度(m)：'  + params.data[0] +
                '<br/>距离(km)： ' + params.data[1] +
                '<br/>有无增益效果：' + res()
            );
        }
    },
    dataZoom:[
        {
            show: true,
            realtime: true,
            height:13,
            type: 'slider',
            xAxisIndex:0,
            handleStyle: {
                color: '#8A2BE2',
            }
        },
        {
            show: true,
            realtime: true,
            width:13,
            type: 'slider',
            yAxisIndex:0,
            left:'3%',
            handleStyle: {
            color: '#8A2BE2',
        }
        },
        {
            type: 'inside',
            xAxisIndex:0,
        },
        {
            type: 'inside',
            yAxisIndex:0,
        }
    ],
    toolbox:{
        show:true,
        itemSize:16,
        feature:{
            dataView:{
                readOnly: true,
                optionToContent: function(opt) {
                    var seriesData = opt.series[0].data;
                    var table = '<table style="width:100%; text-align:center"><tbody><tr>' +
                                '<td>高度(m)</td><td>距离(km)</td><td>有无增益</td>' +
                                '</tr>';
                    for (var i=0, l=seriesData.length; i<l; i++) {
                        let res = function () {
                            if (seriesData[i][2]) return "有";
                            return "无";
                        };
                        table += '<tr>' +
                                 '<td>' + seriesData[i][0] + '</td>' +
                                 '<td>' + seriesData[i][1] + '</td>' +
                                 '<td>' + res() + '</td>' +
                                 '</tr>';
                    }
                    table += '</tbody></table>';
                    return table;
                }
            },
            restore: {},
            saveAsImage: {},
            dataZoom: {},
        },
        left:'center'
    },
    xAxis: {
        type: 'category',
        name:'距离/km',
        nameLocation:'center',
        nameTextStyle:{
            padding:10,
            fontSize:16,
        }
    },
    yAxis: {
        type: 'category',
        name:'高度/m',
        nameLocation:'center',
        nameTextStyle:{
            padding:20,
            fontSize:16,
        }
    },
    visualMap: {
        min: 0,
        max: 1,
        calculable: true,
        realtime: false,
		right:'2%',
        bottom:'3%',
        inRange: {
            color: ['#1C1C1C', '#363636', '#4F4F4F', '#696969', '#828282', '#9C9C9C', '#B5B5B5', '#CFCFCF', '#E8E8E8']
        }
    },
    series: {
        name: '雷达传输增益',
        type: 'heatmap',
        data: radar_data,
        emphasis: {
            itemStyle: {
                borderColor: '#333',
                borderWidth: 1
            }
        },
        progressive: 1000,
        animation: false
    }
};