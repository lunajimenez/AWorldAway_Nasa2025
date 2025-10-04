import "vue-router";

declare module "vue-router" {
    interface RouteMeta extends Components.PageMeta {}
}

declare module "#app" {
    interface PageMeta extends Components.PageMeta {}
}

export {};
