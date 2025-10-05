import type { $Fetch } from "nitropack";
import "vue-router";

declare module "vue-router" {
    interface RouteMeta extends Components.PageMeta {}
}

declare module "#app" {
    interface PageMeta extends Components.PageMeta {}

    interface NuxtApp {
        $fetchRoot: $Fetch;
        $fetchApi: $Fetch;
    }
}

declare module "vue" {
    interface ComponentCustomProperties {
        $fetchRoot: $Fetch;
        $fetchApi: $Fetch;
    }
}

export {};
