var chart3 = echarts.init(document.getElementById("chart3"))
var Perf_T = [0,0,0,0]
var Perf_L = [0,0,0,0]
var option3 = {
    tooltip: {},
    legend: {
        data: ['预测性能平均', '最后一次预测性能']
    },
    radar: {
        // shape: 'circle',
        name: {
            textStyle: {
                color: '#fff',
                backgroundColor: '#999',
                borderRadius: 3,
                padding: [3, 5]
            }
        },
        indicator: [
            { name: '绝对平方误差', max: 20},
            { name: '绝对平方百分比误差(%)', max: 100},
            { name: '平均平方根误差', max: 50},
            { name: '置信程度(%)', max: 100},
        ],
        splitArea: {
                areaStyle: {
                    color: ['rgba(114, 172, 209, 0.2)',
                        'rgba(114, 172, 209, 0.4)', 'rgba(114, 172, 209, 0.6)',
                        'rgba(114, 172, 209, 0.8)', 'rgba(114, 172, 209, 1)'],
                    shadowColor: 'rgba(0, 0, 0, 0.3)',
                    shadowBlur: 10
                }
        },
    },
    series: {
        name: 'result',
        type: 'radar',
        // areaStyle: {normal: {}},
        data: [
            {
                value: Perf_T,
                name: '预测性能平均'
            },
            {
                value: Perf_L,
                name: '最后一次预测性能'
            }
        ]
    }
};
function detail(){
	but_name = document.getElementById("detail-button").innerHTML
	if(but_name=="详细信息"){
		document.getElementById("prediction-next").style.display = "none"
		document.getElementById("prediction-detail").style.display = ""
		document.getElementById("detail-button").innerHTML = "预测信息"
	}
	else{
		document.getElementById("prediction-next").style.display = ""
		document.getElementById("prediction-detail").style.display = "none"
		document.getElementById("detail-button").innerHTML = "详细信息"
	}
}