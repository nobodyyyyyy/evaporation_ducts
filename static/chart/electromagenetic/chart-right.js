var chart_right = echarts.init(document.getElementById("chart-right"));
option_right = {
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
    tooltip: {
        formatter: function (params) {
            return (
                '高度(m)：'  + params.data[0] +
                '<br/>距离(km)： ' + params.data[1] +
                '<br/>电磁波损失(dB)：' + params.data[2]
            );
        }
    },
    toolbox:{
        show:true,
        itemSize:16,
        feature:{
            dataView:{
                readOnly: true,
                optionToContent: function(opt) {
                    let seriesData = opt.series[0].data;
                    let table = '<table style="width:100%; text-align:center"><tbody><tr>' +
                                '<td>高度(m)</td><td>距离(km)</td><td>电磁波传播损失(dB)</td>' +
                                '</tr>';
                    for (let i=0, l=seriesData.length; i<l; i++) {
                        table += '<tr>' +
                                 '<td>' + seriesData[i][0] + '</td>' +
                                 '<td>' + seriesData[i][1] + '</td>' +
                                 '<td>' + seriesData[i][2] + '</td>' +
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
        //data: xData,
        name:'距离/km',
        nameLocation:'center',
        nameTextStyle:{
            padding:10,
            fontSize:16,
        }
    },
    yAxis: {
        type: 'category',
        //data: yData,
        name:'高度/m',
        nameLocation:'center',
        nameTextStyle:{
            padding:20,
            fontSize:16,
        }
    },
    visualMap: {
        min: elec_min,
        max: elec_max,
        calculable: true,
        realtime: false,
		right:'2%',
        bottom:'3%',
        inRange: {
            color: ['#ad3cff', '#0000ff', '#00b2d3',
                '#ccfff1', '#ffe9a4', '#ffbe3f',
                '#ffa742', '#cf512c', '#980603']
        }
    },
    series: {
        name: '电磁波损失',
        type: 'heatmap',
        data: elec_data,
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
option_left.series.data = radar_data;
option_right.series.data = elec_data;
chart_left.setOption(option_left, true);
chart_right.setOption(option_right, true);