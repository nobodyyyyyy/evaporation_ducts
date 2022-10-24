var myChart1 = echarts.init(document.getElementById("chart1"));
var merge_data = [SD, ED]
option1 = {
    tooltip: {
        trigger: 'axis',
        formatter: function (params) {
            params = params[0];
            return (
            '高度：' + params.data[1] + "m" +
            '<br />大气折射指数： ' + params.data[0]
      );
    },
        axisPointer: {
            animation: false
        }
    },
    legend: {
        data: ['大气折射指数'],
        left: 10
    },
    toolbox: {
        feature: {
            dataView:{
                readOnly: true,
                optionToContent: function(opt) {
                    var seriesData = opt.series[0].data;
                    var table = '<table style="width:100%; text-align:center"><tbody><tr>' +
                                '<td>高度(m)</td><td>大气折射指数</td>' +
                                '</tr>';
                    for (var i=0, l=seriesData.length; i<l; i++) {
                        table += '<tr>' +
                                 '<td>' + seriesData[i][1] + '</td>' +
                                 '<td>' + seriesData[i][0] + '</td>' +
                                 '</tr>';
                    }
                    table += '</tbody></table>';
                    return table;
                }
            },
            dataZoom: {
                yAxisIndex: 'none'
            },
            magicType: {type: ['line', 'bar']},
            restore: {},
            saveAsImage: {}
        }
    },

    xAxis: [
        {
            name: 'M-units',
            type: 'value',
            boundaryGap: false,
            axisLine: {onZero: true},
        },
    ],
    yAxis: [
        {
            name: '高度(m)',
            type: 'value',
        }
    ],

    series: [
        {
            name: '大气折射指数',
            type: 'line',
            hoverAnimation: false,
            data: Refraction,
            itemStyle: {
            	normal:{
            		 color: '#00CED1'
            	}
            },
        },
    ]
};
