// Declaration for Node.js process.env in Convex actions
declare const process: {
  env: {
    MODAL_ENDPOINT_URL?: string
    CONVEX_SITE_URL?: string
    [key: string]: string | undefined
  }
}
