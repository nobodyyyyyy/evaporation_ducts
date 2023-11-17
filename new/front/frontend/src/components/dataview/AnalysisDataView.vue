<template>
  <div id="wrapper">
    <el-container class="container" style="height: 100%">
      <el-header class="navi_header">
          <el-row>
            <el-col :span="24">
              <el-breadcrumb separator-class="el-icon-arrow-right" class="breadcrumb-wrapper child">
                <!--          <el-breadcrumb-item :to="{ path: '/' }">首页</el-breadcrumb-item>-->
                <el-breadcrumb-item>数据展示</el-breadcrumb-item>
                <el-breadcrumb-item>再分析资料</el-breadcrumb-item>
              </el-breadcrumb>
            </el-col>
          </el-row>
          <el-row class="title-row">
            <el-col :span="24"><span class="title-text">再分析资料可视化展示</span></el-col>
          </el-row>
      </el-header>
      <el-main>
        <div class="search-box">
          <div class="search-row">
            <span class="search-text">数据来源</span>
            <template>
              <el-select v-model="sourceSelected" placeholder="请选择" @change="sourceSelectedChangeEvent">
                <el-option
                  v-for="item in sourceOptions"
                  :key="item.value"
                  :label="item.label"
                  :value="item.value">
                </el-option>
              </el-select>
            </template>
            <i class="el-icon-info select-icon"></i>
            <span class="hint-text">数据来源即再分析资料的采集方</span>
          </div>
          <div class="search-row">
            <span class="search-text">数据类型</span>
            <template>
              <el-select v-model="typeSelected" placeholder="请选择" @change="typeSelectedChangeEvent"
                         no-data-text="请先选择数据来源">
                <el-option
                  v-for="item in typeOptions"
                  :key="item.value"
                  :label="item.label"
                  :value="item.value">
                  <span style="float: left">{{ item.label }}</span>
                  <span style="float: right; color: #8492a6; font-size: 13px">{{ item.value }}</span>
                </el-option>
              </el-select>
            </template>
            <i class="el-icon-info select-icon"></i>
            <span class="hint-text">数据类型即再分析资料的类别，例如海表温度、风速等</span>
          </div>

          <div class="search-row" v-show="levelShown && displayModeSelected !== '垂直廓线显示'">
            <span class="search-text">标准大气 level </span>
            <template>
              <el-select v-model="levelSelected" placeholder="请选择" @change="typeSelectedChangeEvent"
                         no-data-text="请先选择数据来源">
                <el-option
                  v-for="item in levelOptions"
                  :key="item.value"
                  :label="item.label"
                  :value="item.value">
<!--                  <span style="float: left">{{ item.label }}</span>-->
<!--                  <span style="float: right; color: #8492a6; font-size: 13px">{{ item.value }}</span>-->
                </el-option>
              </el-select>
            </template>
            <i class="el-icon-info select-icon"></i>
            <span class="hint-text">Major level type indicator，诸如比湿度、U风分量、V风分量等数据需要选择展示的层值</span>
          </div>

          <div class="search-row">
            <span class="search-text">显示方式</span>
            <el-radio-group v-model="displayModeSelected" size="medium" @change="onRangeSelectBtnChangeListener">
              <el-radio-button v-for="mode in displayModes" :label="mode" :key="mode"/>
            </el-radio-group>
<!--            <i class="el-icon-info select-icon"></i>-->
<!--            <span class="hint-text">您可以查看某一天的具体数值，也可以查看一段时间内的变化趋势</span>-->
            <span class="search-text">日期选择</span>
            <el-date-picker :class="{'single-date-select': true}"
                            v-model="singleDateSelect" type="date"
                            :placeholder=dateSelectPlaceHolder
                            :readonly=dateSelectReadOnly
                            :disabled=dateSelectReadOnly
                            @change="singleDateSelectEvent" size="small"
                            :picker-options="dateOption"
                            :default-value="dateFrom"
                            value-format="timestamp"
                            v-show="displayModeSelected === '按天显示' || displayModeSelected === '垂直廓线显示'"/>
<!--            周期显示选择时间段-->
            <el-date-picker
              v-show="displayModeSelected === '按时间段显示'"
              v-model="dateRangeSelected"
              type="daterange"
              :picker-options="dateOption"
              :readonly=dateSelectReadOnly
              :disabled=dateSelectReadOnly
              :default-value="dateFrom"
              value-format="timestamp"
              size="small"
              range-separator="至"
              start-placeholder="开始日期"
              end-placeholder="结束日期"/>
          </div>
          <div class="search-row" style="display: flex; margin-top: 8px">
            <span class="search-text" v-show="displayModeSelected === '按天显示'">纬度区间</span>
            <span class="search-text" v-show="displayModeSelected !== '按天显示'">纬度选择</span>
            <span class="latlng-hint-text">南纬90°</span>
            <template>
              <div class="latlng_slider" v-show="displayModeSelected === '按天显示'">
                <el-slider v-model="latSelected" range :max="90" :min="-90"></el-slider>
              </div>
              <div class="latlng_slider" v-show="displayModeSelected !== '按天显示'">
                <el-slider v-model="singleLatSelected" :max="90" :min="-90"></el-slider>
              </div>
            </template>
            <span class="latlng-hint-text">北纬90°</span>
          </div>
          <div class="search-row" style="display: flex; margin-top: 8px">
            <span class="search-text" v-show="displayModeSelected === '按天显示'">经度区间</span>
            <span class="search-text" v-show="displayModeSelected !== '按天显示'">经度选择</span>
            <span class="latlng-hint-text">西经180°</span>
            <template>
              <div class="latlng_slider" v-show="displayModeSelected === '按天显示'">
                <el-slider v-model="lngSelected" range :max="180" :min="-180"></el-slider>
              </div>
              <div class="latlng_slider single" v-show="displayModeSelected !== '按天显示'">
                <el-slider v-model="singleLngSelected" :max="180" :min="-180"></el-slider>
              </div>
            </template>
            <span class="latlng-hint-text">东经180°</span>
          </div>
          <div class="search-row">
            <el-button class="search-btn" round v-on:click="searchForSingleDate">搜索并显示</el-button>
          </div>
        </div>
        <div class="display-box" v-show=resultShown v-loading="loading">
          <div id="heatmapChart" style='width: 1200px; height: 800px; overflow: auto; left: 50px' v-if="heatmapShown"></div>
          <div id="lineChart" style='width: 1200px; height: 800px; overflow: auto; left: 50px' v-if="lineChartShown"></div>
          <div id="singleDayLineChart" style='width: 1200px; height: 800px; overflow: auto; left: 50px' v-if="singleDayLineChartShown"></div>
        </div>
      </el-main>
    </el-container>
  </div>
</template>

<script>

import SideMenu from './SideMenu'

export default {
  name: 'AnalysisDataView',
  components: {SideMenu},
  data () {
    return {
      nameMapping: {},
      info: {},
      typeOptions: [],
      tempTypeOptions: [],
      typeSelected: '',
      sourceOptions: [],
      sourceSelected: '',
      displayModes: ['按天显示', '按时间段显示', '垂直廓线显示'],
      displayModeSelected: '按天显示',
      singleDateSelect: '',
      dateSelectPlaceHolder: '请先选择数据来源和类型',
      dateSelectReadOnly: true,
      dateFrom: '',
      dateTo: '',
      dateRangeSelected: '',
      dateOption: {
        disabledDate: (time) => {
          const _this = this
          return time >= _this.dateTo || time <= _this.dateFrom
        }
      },
      singleLatSelected: 0,
      singleLngSelected: 0,
      latSelected: [-45, 45],
      lngSelected: [-90, 90],
      levelOptions: [],
      // 什么样的数据需要选择 level
      levelShownTypes: [],
      levelShownOptions: [],
      // --
      levelSelected: '',
      levelShown: false,
      resultShown: false,
      loading: false,
      heatmapShown: false,
      heatMapExist: false,
      lineChartShown: false,
      lineChartExist: false,
      // 垂直廓线显示
      singleDayLineChartShown: false,
      singleDayLineChartExist: false
    }
  },
  mounted () {
    this.initEntries()
  },
  methods: {
    initEntries () {
      const _this = this
      const api = 'data/analysis/init'
      this.$axios.get(api).then(resp => {
        if (resp && resp.status === 200) {
          // 1. 记录中英文映射表
          let mappingEN = resp.data.mapping_eng
          let mappingCN = resp.data.mapping_cn
          _this.typeOptions = []
          _this.levelShownOptions = []
          for (let i = 0; i < mappingCN.length; ++i) {
            _this.nameMapping[mappingEN[i]] = mappingCN[i]
          }
          // 加载层级信息
          _this.levelOptions = []
          const levels = resp.data.level.reverse()
          for (let i = 0; i < levels.length; ++i) {
            _this.levelOptions.push({value: levels[i], label: levels[i], key: levels[i]})
          }
          _this.levelShownTypes = resp.data.need_show_level

          // 2. 记录不同数据的配置信息
          let data = resp.data.data
          for (let i = 0; i < data.length; i++) {
            const cur = data[i]
            _this.info[cur['source']] = cur
            _this.sourceOptions.push({value: cur['source'], label: cur['source'], key: cur['source']})
          }
        }
      })
      // 测试区域
      // const d = '2018-01-01'
      // console.log(new Date(d).getTime())
      // console.log(Date.now())
    },
    typeSelectedChangeEvent (val) {
      // 数据类型选择更变
      const _this = this
      // 获取可以选择的日期范围（和高度level）
      const source = _this.sourceSelected
      const typeInfo = _this.info[source][val]
      _this.dateFrom = new Date(typeInfo['date_from']).getTime()
      _this.dateTo = new Date(typeInfo['date_to']).getTime()
      _this.singleDateSelectPlaceHolder = '请选择日期'
      _this.dateSelectReadOnly = false
      _this.singleDateSelect = ''
      _this.dateRangeSelected = ''
      _this.levelShown = _this.levelShownTypes.indexOf(val) !== -1
    },
    sourceSelectedChangeEvent (val) {
      // 数据来源选择更变
      const _this = this
      const types = _this.info[val]['types']
      _this.typeOptions = []
      _this.levelShownOptions = []
      _this.singleDateSelectPlaceHolder = '请先选择数据来源和类型'
      _this.singleDateSelect = ''
      _this.dateRangeSelected = ''
      _this.typeSelected = ''
      _this.dateFrom = ''
      _this.dateTo = ''
      _this.dateSelectReadOnly = true
      _this.levelShown = false
      if (_this.displayModeSelected === '垂直廓线显示') {
        for (let i = 0; i < types.length; ++i) {
          if (_this.levelShownTypes.indexOf(types[i]) !== -1) {
            _this.typeOptions.push({value: types[i], label: _this.nameMapping[types[i]], key: types[i]})
          }
        }
      } else {
        for (let i = 0; i < types.length; ++i) {
          _this.typeOptions.push({value: types[i], label: _this.nameMapping[types[i]], key: types[i]})
        }
      }
    },
    singleDateSelectEvent (val) {
      // 单天日期选择事件
    },
    onRangeSelectBtnChangeListener () {
      // 范围选择按钮点击后清空已选
      let _this = this
      // if (_this.displayModeSelected === '按时间段显示') {
      //   _this.singleLatSelected = 0
      //   _this.singleLngSelected = 0
      // } else {
      //   _this.singleLatSelected = [-45, 45]
      //   _this.singleLngSelected = [-90, 90]
      // }
      _this.dateRangeSelected = ''
      _this.singleDateSelect = ''
      // if (_this.displayModeSelected === '垂直廓线显示') {
      //   _this.sourceSelected = ''
      //   _this.typeSelected = ''
      //   _this.tempTypeOptions = _this.typeOptions
      //   _this.typeOptions = []
      //   _this.typeOptions = _this.levelShownOptions
      //   console.log(_this.typeOptions)
      //   console.log(_this.levelShownOptions)
      // } else {
      //   if (_this.typeOptions.length !== 0) {
      //     _this.typeOptions = _this.tempTypeOptions
      //   }
      // }
      _this.sourceSelected = ''
      _this.typeOptions = []
      _this.levelShownOptions = []
      _this.singleDateSelectPlaceHolder = '请先选择数据来源和类型'
      _this.singleDateSelect = ''
      _this.dateRangeSelected = ''
      _this.typeSelected = ''
      _this.dateFrom = ''
      _this.dateTo = ''
      _this.dateSelectReadOnly = true
      _this.levelShown = false
    },
    searchForSingleDate () {
      // 主要方法，请求后端数据并在前端展示
      const _this = this
      // 检查数据是否选择完全
      let complete = true
      if (_this.typeSelected === '' || _this.sourceSelected === '') {
        complete = false
      }
      if (_this.displayModeSelected === '按天显示' && _this.singleDateSelect === '') {
        complete = false
      }
      if (_this.displayModeSelected === '按时间段显示' && _this.dateRangeSelected === '') {
        complete = false
      }
      if (_this.levelShown && _this.levelSelected === '' && _this.displayModeSelected !== '垂直廓线显示') {
        complete = false
      }
      if (!complete) {
        this.$message({
          showClose: true,
          message: '条件未输入完全',
          type: 'error'
        })
        return
      }
      _this.resultShown = true
      _this.loading = true
      let api = ''
      let data = {}
      console.log(_this.lng_range)
      if (_this.displayModeSelected === '按时间段显示') {
        api = 'data/analysis/fetch-date-range'
        data = {
          lat: _this.singleLatSelected,
          lng: _this.singleLngSelected,
          type: _this.typeSelected,
          source: _this.sourceSelected,
          timestamp: _this.dateRangeSelected,
          level: _this.levelSelected
        }
      } else if (_this.displayModeSelected === '按天显示') {
        api = 'data/analysis/fetch-single-date'
        data = {
          lat_range: _this.latSelected,
          lng_range: _this.lngSelected,
          type: _this.typeSelected,
          source: _this.sourceSelected,
          timestamp: _this.singleDateSelect,
          level: _this.levelSelected
        }
      } else {
        // 垂直廓线显示
        api = 'data/analysis/fetch-level-daily'
        data = {
          lat: _this.singleLatSelected,
          lng: _this.singleLngSelected,
          type: _this.typeSelected,
          source: _this.sourceSelected,
          timestamp: _this.singleDateSelect,
          level: -1
        }
      }
      console.log(data)
      this.$axios
        .post(api, data)
        .then(successResponse => {
          console.log(successResponse)
          if (_this.displayModeSelected === '按时间段显示') {
            _this.heatmapShown = false
            _this.lineChartShown = true
            _this.singleDayLineChartShown = false
            _this.clearAllDrawings()
            this.$nextTick(() => {
              this.drawRangeLineChart(successResponse)
              _this.lineChartExist = true
            })
            // this.drawRangeLineChart(successResponse)
          } else if (_this.displayModeSelected === '按天显示') {
            _this.heatmapShown = true
            _this.lineChartShown = false
            _this.singleDayLineChartShown = false
            _this.clearAllDrawings()
            this.$nextTick(() => {
              this.drawSingleDayHeatMap(successResponse)
              _this.heatMapExist = true
            })
            // this.drawSingleDayHeatMap(successResponse)
          } else {
            // 垂直廓线显示
            _this.heatmapShown = false
            _this.lineChartShown = false
            _this.singleDayLineChartShown = true
            _this.clearAllDrawings()
            this.$nextTick(() => {
              this.drawSingleDayLineChart(successResponse)
              _this.singleDayLineChartShown = true
            })
          }
          _this.loading = false
        })
        .catch(failResponse => {
          this.$message({
            showClose: true,
            message: '后端出错',
            type: 'error'
          })
          _this.resultShown = true
          _this.loading = false
        })
    },
    clearAllDrawings () {
      const _this = this
      if (_this.heatMapExist) {
        let hmChart = this.$echarts.init(document.getElementById('heatmapChart'))
        hmChart.clear()
        _this.heatMapExist = false
      }
      if (_this.lineChartExist) {
        let lChart = this.$echarts.init(document.getElementById('lineChart'))
        lChart.clear()
        _this.lineChartExist = false
      }
      if (_this.singleDayLineChartExist) {
        let lChart = this.$echarts.init(document.getElementById('singleDayLineChart'))
        lChart.clear()
        _this.singleDayLineChartExist = false
      }
    },
    drawSingleDayHeatMap (successResponse) {
      const lats = successResponse.data.lat
      const lngs = successResponse.data.lng
      const data = successResponse.data.data
      const max = successResponse.data.max_value
      const min = successResponse.data.min_value
      const titleText = successResponse.data.title
      // 测试
      // 基于刚刚准备好的 DOM 容器，初始化 EChart 实例
      let hmChart = this.$echarts.init(document.getElementById('heatmapChart'))
      // 绘制图表
      // prettier-ignore
      const d = data
        .map(function (item) {
          return [item[1], item[0], item[2] || '-']
        })
      console.log(d)
      // console.log(d)
      hmChart.setOption({
        title: {
          text: titleText,
          left: 'center',
          top: '4%'
        },
        tooltip: {
          position: 'top',
          axisPointer: {
            // 坐标轴指示器，坐标轴触发有效
            type: 'shadow' // 默认为直线，可选为：'line' | 'shadow'
          },
          // formatter: '经度：{b}<br/>纬度：{c[0]}<br/>值：{c1}'
          formatter: function (params) {
            const data = params.data
            return '经度: ' + lngs[data[0]] + '<br/>纬度: ' + lats[data[1]] + '<br/>值: ' + data[2]
          }
        },
        grid: {
          height: '50%',
          top: '10%'
        },
        xAxis: {
          name: '经度',
          type: 'category',
          data: lngs,
          splitArea: {
            show: true
          }
        },
        yAxis: {
          name: '纬度',
          type: 'category',
          data: lats,
          splitArea: {
            show: true
          }
        },
        visualMap: {
          min: min,
          max: max,
          calculable: true,
          orient: 'horizontal',
          left: 'center',
          bottom: '25%',
          precision: 5,
          itemWidth: '20',
          itemHeight: hmChart.getWidth() - 400,
          inRange: {
            color: ['#438cf8', '#f8f3a3', '#ff0000'] // 修改热力图的颜色 淡蓝色=>深蓝色的过度
          }
        },
        series: [
          {
            type: 'heatmap',
            data: d,
            label: {
              show: false
            },
            emphasis: {
              itemStyle: {
                shadowBlur: 10,
                shadowColor: 'rgba(0, 0, 0, 0.5)'
              }
            }
          }
        ],
        toolbox: {
          show: true,
          left: '4%',
          top: '2%',
          feature: {
            saveAsImage: {
              show: true,
              excludeComponents: ['toolbox'],
              pixelRatio: 2
            },
            dataZoom: {
              show: true
            }
          }
        }
      })
    },
    drawRangeLineChart (successResponse) {
      const data = successResponse.data.data
      const axis = successResponse.data.time_axis
      const max = successResponse.data.max_value
      const min = successResponse.data.min_value
      const titleText = successResponse.data.title
      // 测试
      // 基于刚刚准备好的 DOM 容器，初始化 EChart 实例
      let lChart = this.$echarts.init(document.getElementById('lineChart'))
      // console.log(axis)
      // console.log(data)
      // 绘制图表
      lChart.setOption({
        title: {
          text: titleText,
          left: 'center',
          top: '4%'
        },
        xAxis: {
          type: 'category',
          name: '经度',
          data: axis
        },
        yAxis: {
          type: 'value',
          name: '纬度',
          min: min,
          max: max
        },
        series: [
          {
            data: data,
            type: 'line',
            symbolSize: 10,
            symbol: 'circle',
            smooth: true,
            lineStyle: {
              width: 3,
              shadowColor: 'rgba(0,0,0,0.3)',
              shadowBlur: 10,
              shadowOffsetY: 8
            }
            // smooth: true
          }
        ],
        tooltip: {
          position: 'top'
        },
        toolbox: {
          show: true,
          left: '4%',
          top: '2%',
          feature: {
            saveAsImage: {
              show: true,
              excludeComponents: ['toolbox'],
              pixelRatio: 2
            },
            dataZoom: {
              show: true
            }
          }
        }
      })
    },
    drawSingleDayLineChart (successResponse) {
      const data = successResponse.data.values
      const heights = successResponse.data.heights
      const valueMinMax = successResponse.data.value_min_max
      // const heightMinMax = successResponse.data.height_min_max
      const titleText = successResponse.data.title
      const unit = successResponse.data.unit
      const desp = successResponse.data.cn_desp
      let lChart = this.$echarts.init(document.getElementById('singleDayLineChart'))
      console.log(heights)
      console.log(data)
      // 绘制图表
      lChart.setOption({
        title: {
          text: titleText,
          left: 'center',
          top: '5%'
        },
        // legend: {
        //   data: ['Altitude (km) vs. temperature (°C)']
        // },
        tooltip: {
          trigger: 'axis',
          formatter: desp + '<br/>{b}m : {c}' + unit
        },
        toolbox: {
          show: true,
          left: '4%',
          top: '2%',
          feature: {
            saveAsImage: {
              show: true,
              excludeComponents: ['toolbox'],
              pixelRatio: 2
            },
            dataZoom: {
              show: true
            }
          }
        },
        grid: {
          height: '80%',
          top: '10%',
          containLabel: true
        },
        xAxis: {
          type: 'value',
          axisLabel: {
            formatter: '{value} ' + unit
          },
          min: valueMinMax[0],
          max: valueMinMax[1]
        },
        yAxis: {
          type: 'category',
          axisLine: { onZero: false },
          axisLabel: {
            formatter: '{value} m'
          },
          boundaryGap: false,
          data: heights
        },
        series: [
          {
            name: titleText,
            type: 'line',
            symbolSize: 10,
            symbol: 'circle',
            smooth: true,
            lineStyle: {
              width: 3,
              shadowColor: 'rgba(0,0,0,0.3)',
              shadowBlur: 10,
              shadowOffsetY: 8
            },
            data: data
          }
        ]
      })
    },
    dummy () {
      return null
    }
  }
}
</script>

<style scoped>
#wrapper {
  height: 95%;
}

.navi_header {
  background-color: white;
  color: #333;
  height: 90px !important;
  border-radius: 10px;
}

.el-main {
  background-color: #E9EEF3;
  color: #333;
  text-align: center;
  //line-height: 160px;
}

body > .el-container {
  margin-bottom: 40px;
}

.breadcrumb-wrapper {
  margin-top: 20px;
  width: 100%;
}

.el-col {
  text-align: left;
}

.title-row {
  height: 20px;
  text-align: left;
  margin-top: 8px;
}

.title-text {
  text-align: left;
  margin-right: 12px;
  margin-bottom: 0;
  color: rgba(0,0,0,.85);
  font-weight: 600;
  font-size: 20px;
  line-height: 32px;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.hint-text {
  color: rgba(98, 97, 97, 0.63);
  text-overflow: ellipsis;
  font-size: 14px;
}

.latlng-hint-text {
  color: rgba(98, 97, 97, 0.63);
  text-overflow: ellipsis;
  font-size: 14px;
  margin-left: 12px;
  margin-right: 12px;
}

.search-box {
  margin-left: -20px;
  margin-right: -20px;
  min-height: 370px;
  background-color: white;
  border-radius: 10px;

}

.display-box {
  margin-left: -20px;
  margin-right: -20px;
  margin-top: 20px;
  min-height: 170px;
  background-color: white;
  border-radius: 10px;
  overflow: auto;
}

.search-row {
  text-align: left;
  padding-top: 20px;
}

.search-text  {
  color: rgba(0,0,0,.85);
  font-weight: 500;
  font-size: 16px;
  margin-left: 23px;
  margin-right: 23px;
}

.select-icon {
  margin-left: 20px;
  margin-right: 10px;
}

.forbidden {
  cursor: no-drop;
}

.latlng_slider {
  width: 400px;
  margin-top: -7px;
}

.search-btn {
  margin-left: 20px;
  margin-bottom: 20px;
}

.el-slider__bar {
  background-color: #E4E7ED !important;
}

// 不显示滚动条
::-webkit-scrollbar {
  width: 0 !important;
}
::-webkit-scrollbar {
  width: 0 !important;height: 0;
}

// 测试 sty
.bg-purple-dark {
  background: #99a9bf;
}
.bg-purple {
  background: #d3dce6;
}
.bg-purple-light {
  background: #e5e9f2;
}
.grid-content {
  border-radius: 4px;
  min-height: 36px;
}

</style>
