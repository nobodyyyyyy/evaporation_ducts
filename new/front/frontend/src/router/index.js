import Vue from 'vue'
import Router from 'vue-router'
import AppIndex from '../home/AppIndex'
import Login from '../components/Login'
import Home from '../components/Home'
import AnalysisDataView from '../components/dataview/AnalysisDataView.vue'
import OriginDataView from '../components/dataview/OriginDataView.vue'
import Main from '../components/Main.vue'

Vue.use(Router)

export default new Router({
  mode: 'history',
  routes: [
    {
      path: '/login',
      name: 'Login',
      component: Login
    },
    {
      path: '/home',
      name: 'Home',
      component: Home,
      // home页面并不需要被访问
      redirect: '/index',
      children: [
        {
          path: '/main',
          name: 'Main',
          component: Main
        },
        {
          path: '/index',
          name: 'AppIndex',
          component: AppIndex
        },
        {
          path: '/AnalysisDataView',
          name: 'AnalysisDataView',
          component: AnalysisDataView
        },
        {
          path: '/OriginDataView',
          name: 'OriginDataView',
          component: OriginDataView
        }
      ]
    },
    {
      path: '/login',
      name: 'Login',
      component: Login
    }
  ]
})
