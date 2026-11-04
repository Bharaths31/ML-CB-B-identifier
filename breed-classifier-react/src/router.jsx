import { createBrowserRouter, createHashRouter, RouterProvider } from "react-router-dom";
import UserPage from "./pages/UserPage.jsx";
import DevPage from "./pages/DevPage.jsx";
import AndroidUserPage from "./pages/AndroidUserPage.jsx";
import { IS_NATIVE } from "./platform.js";

const webRouter = createBrowserRouter([
  {
    path: "/",
    element: <UserPage />,
  },
  {
    path: "/dev",
    element: <DevPage />,
  },
]);

// Single-portal native app: no /dev navigation, all advanced controls live in
// the in-page Settings drawer.
const nativeRouter = createHashRouter([
  {
    path: "/",
    element: <AndroidUserPage />,
  },
  {
    path: "*",
    element: <AndroidUserPage />,
  },
]);

export default function AppRouter() {
  return <RouterProvider router={IS_NATIVE ? nativeRouter : webRouter} />;
}
