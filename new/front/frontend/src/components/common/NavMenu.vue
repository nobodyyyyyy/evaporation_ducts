<template>
  <el-menu
    router
    mode="vertical"
    :default-active="'/main'"
    text-color="black"
    background-color=""
    >
<!--    <el-menu-item v-for="(item, i) in navList" :key="i" :index="item.name">-->
<!--      {{item.navItem}}-->
<!--    </el-menu-item>-->
    <el-menu-item index="/main" :key="1">
      <i class="el-icon-s-home"></i>
      <span>首页</span>
    </el-menu-item>
    <el-submenu index="2">
      <template class="item-entry" slot="title">
        <i class="el-icon-s-data"></i>
        <span>数据展示</span>
      </template>
      <el-menu-item index="OriginDataView">探空资料</el-menu-item>
      <el-menu-item index="AnalysisDataView">再分析资料</el-menu-item>
    </el-submenu>
    <el-submenu index="/index" :key="3">
      <template class="item-entry" slot="title">
        <i class="el-icon-data-line"></i>
        <span>大气波导</span>
      </template>
      <el-menu-item index="OriginDataView">波导计算</el-menu-item>
      <el-menu-item index="AnalysisDataView">波导预测</el-menu-item>
    </el-submenu>
    <el-menu-item index="/admin" :key="4">
      <i class="el-icon-stopwatch"></i>
      <span>波导模拟</span>
    </el-menu-item>
<!--    <a href="#nowhere" style="color: #222;float: right;padding: 20px;">更多功能</a>-->
<!--    <i class="el-icon-menu" style="float:right;font-size: 45px;color: #222;padding-top: 8px"></i>-->
<!--    <span style="position: absolute;padding-top: 20px;right: 46%;font-size: 23px;font-weight: bold">短临预测蒸发波导软件</span>-->
  </el-menu>
</template>

<script>
export default {
  name: 'NavMenu',
  data () {
    return {
      navList: [
        {name: '/main', navItem: '首页'},
        {name: '/data-view', navItem: '数据展示'},
        {name: '/index', navItem: '波导预测'},
        {name: '/admin', navItem: '传输损耗'}
      ]
    }
  }
}
</script>

<style scoped>
a{
  text-decoration: none;
}

span {
  pointer-events: none;
}

.el-menu {
  text-align: left;
  margin-left: 5px;
}

// 不显示滚动条
::-webkit-scrollbar {
  width: 0 !important;
}
::-webkit-scrollbar {
  width: 0 !important;height: 0;
}
.inner::-webkit-scrollbar {
  display: none;
}

</style>
