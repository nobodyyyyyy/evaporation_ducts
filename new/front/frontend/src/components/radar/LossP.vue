<template>
  <div id="wrapper">
    <el-container class="container" style="height: 100%">
      <el-header class="navi_header">
        <el-row>
          <el-col :span="24">
            <el-breadcrumb separator-class="el-icon-arrow-right" class="breadcrumb-wrapper child">
              <!--          <el-breadcrumb-item :to="{ path: '/' }">首页</el-breadcrumb-item>-->
              <el-breadcrumb-item>波导模拟</el-breadcrumb-item>
              <el-breadcrumb-item>虚警率计算</el-breadcrumb-item>
            </el-breadcrumb>
          </el-col>
        </el-row>
        <el-row class="title-row">
          <el-col :span="24"><span class="title-text">大气波导传输损耗相关虚警率计算</span></el-col>
        </el-row>
      </el-header>
      <el-main>
        <div class="search-box">
          <!--          t = data['temp']  # 气温-->
          <!--          eh = data['relh']  # 相对湿度-->
          <!--          u = data['speed']  # 风速-->
          <!--          p = data['pressure']  # 压强-->
          <!--          h = data['height']  # 测量高度-->
          <!--          sst = data['sst']  # 海面温度-->
          <div class="search-row">
            <span class="search-text">虚警概率</span>
            <el-input placeholder="请输入虚警概率" v-model="pfa">
              <template slot="append">范围(0,1)</template>
            </el-input>
            <span class="search-text">横截面积</span>
            <el-input placeholder="请输入横截面积" v-model="sigma">
              <template slot="append">m * m</template>
            </el-input>
          </div>
          <div class="search-row">
            <el-button class="search-btn" round v-on:click="onCalBtnClicked">计算</el-button>
            <i class="el-icon-info select-icon"></i>
            <span class="hint-text">计算结果不会保存</span>
          </div>
        </div>
        <div class="display-box">
          <el-row class="title-row" style=" margin-top: 15px;">
            <el-col :span="24"><span class="title-text" style="padding-left: 20px; ">历史计算结果</span></el-col>
          </el-row>
          <template>
            <el-table
              class="result-table" :data="tableData" stripe style="width: 100%">
              <el-table-column prop="date" label="计算时间"></el-table-column>
              <el-table-column prop="pfa" label="虚警概率"></el-table-column>
              <el-table-column prop="sigma" label="横截面积"></el-table-column>
              <el-table-column prop="pdetect" label="探测概率"></el-table-column>
            </el-table>
          </template>
        </div>
      </el-main>
    </el-container>
  </div>
</template>

<script>
export default {
  name: 'LossP',
  components: {},
  data () {
    return {
      pfa: '',
      sigma: '',
      tableData: []
    }
  },
  mounted () {

  },
  methods: {
    onCalBtnClicked () {
      const _this = this
      const api = 'radar/p_detect_cal'
      const data = {
        'pfa': _this.pfa,
        'sigma': _this.sigma
      }
      this.$axios
        .post(api, data)
        .then(successResponse => {
          const code = successResponse.data.code
          const res = successResponse.data.res
          if (code === 0) {
            this.$notify({
              title: '计算完成',
              message: '成功计算得到波导高度为：' + res,
              type: 'success'
            })
            _this.tableData.unshift(successResponse.data.table_entry)
          } else if (code === -1) {
            this.$notify({
              title: '结果出错',
              message: successResponse.data.msg,
              type: 'warning'
            })
          }
        })
        .catch(failResponse => {
          _this.loading = false
          console.log(failResponse)
          this.$notify({
            title: '无法计算',
            message: '后端出错',
            type: 'error'
          })
        })
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
  min-height: 140px;
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

.el-input {
  width: 250px;
}
.input-with-select .el-input-group__prepend {
  background-color: #fff;
}

.search-btn {
  width: 150px;
  margin-left: 20px;
  margin-bottom: 20px;
}
.result-table {
  padding: 20px 0 20px 20px;
}
// 不显示滚动条
::-webkit-scrollbar {
  width: 0 !important;
}
::-webkit-scrollbar {
  width: 0 !important; height: 0;
}

</style>
