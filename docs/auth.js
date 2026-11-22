/**
 * Demo version of the dashboard sign-in: the same role + 4-digit PIN screen as production, checked in the browser
 * against published demo PINs (see the "Demo PINs" button). Same API as the production module.
 */
var DemoAuth = (function () {
  var SESSION_KEY = 'northwind_session';
  var ROLES = [
    { value: 'evp', label: 'EVP', pin: '1111', team: null },
    { value: 'admin', label: 'Admin / Ops', pin: '2222', team: null },
    { value: 'manager_central', label: 'Central Projects Manager', pin: '3001', team: 'Central Projects' },
    { value: 'manager_acscr', label: 'DLR - CR Manager', pin: '3002', team: 'DLR - CR' },
    { value: 'manager_west', label: 'West Projects Manager', pin: '3003', team: 'West Projects' },
    { value: 'manager_acswr', label: 'DLR - WR Manager', pin: '3004', team: 'DLR - WR' },
    { value: 'manager_acser_east', label: 'DLR - ER & East Projects Manager', pin: '3005', team: ['DLR - ER', 'East Projects'] },
    { value: 'manager_direct', label: 'Direct Channel Manager', pin: '3006', team: 'Direct Channel' },
    { value: 'se', label: 'Sales Engineer', pin: '4000', team: null },
  ];
  var FULL_ACCESS = ['admin', 'evp'];
  var MANAGERS = ['manager_central', 'manager_acscr', 'manager_west', 'manager_acswr', 'manager_acser_east', 'manager_direct'];
  var TAB_ACCESS = { redlist: ['admin', 'evp'].concat(MANAGERS), evp: ['admin', 'evp'], inter: ['admin', 'evp'],
                     openbox: ['admin', 'evp', 'se'].concat(MANAGERS) };
  var _session = null, _onLogin = null;

  function _load() { try { var r = sessionStorage.getItem(SESSION_KEY); if (r) _session = JSON.parse(r); } catch (e) { _session = null; } }
  function _save(s) { _session = s; try { sessionStorage.setItem(SESSION_KEY, JSON.stringify(s)); } catch (e) {} }
  function _clear() { _session = null; try { sessionStorage.removeItem(SESSION_KEY); } catch (e) {} }

  function pinTable() {
    return '<table style="border-collapse:collapse;width:100%;font-size:11.5px;text-align:left">' +
      '<tr><th style="padding:4px 6px;border-bottom:1px solid #e5e7eb;color:#6b7280">Role</th><th style="padding:4px 6px;border-bottom:1px solid #e5e7eb;color:#6b7280">PIN</th><th style="padding:4px 6px;border-bottom:1px solid #e5e7eb;color:#6b7280">Sees</th></tr>' +
      ROLES.map(function (r) {
        var sees = r.value === 'evp' || r.value === 'admin' ? 'All teams, all tabs' : r.value === 'se' ? 'Release Pool; no AR' : (Array.isArray(r.team) ? r.team.join(' + ') : r.team) + ' only';
        return '<tr><td style="padding:4px 6px;border-bottom:1px solid #f1f3f5">' + r.label + '</td><td style="padding:4px 6px;border-bottom:1px solid #f1f3f5;font-weight:600;font-family:monospace">' + r.pin + '</td><td style="padding:4px 6px;border-bottom:1px solid #f1f3f5;color:#6b7280">' + sees + '</td></tr>';
      }).join('') + '</table>' +
      '<div style="font-size:11px;color:#6b7280;margin-top:8px;line-height:1.5">The same PINs open the locked tabs inside the reports ' +
      '(Release Pool watch list: a manager PIN unlocks that team, Admin unlocks all; EVP analytics and inter-team: Admin).</div>';
  }

  function helpButton() {
    if (document.getElementById('demo-pin-help')) return;
    var b = document.createElement('button');
    b.id = 'demo-pin-help';
    b.textContent = '\u{1F464} View as: ' + (_session && _session.label ? _session.label : 'EVP');
    b.style.cssText = 'position:fixed;right:14px;bottom:14px;z-index:100000;background:#111827;color:#fff;border:0;border-radius:20px;' +
      'padding:8px 13px;font:600 12px "DM Sans",-apple-system,Segoe UI,sans-serif;cursor:pointer;box-shadow:0 4px 14px rgba(0,0,0,.2)';
    var p = document.createElement('div');
    p.style.cssText = 'position:fixed;right:14px;bottom:56px;z-index:100000;background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:14px;' +
      'width:340px;max-width:calc(100vw - 28px);box-shadow:0 10px 30px rgba(0,0,0,.15);display:none;font-family:"DM Sans",-apple-system,Segoe UI,sans-serif;color:#111827';
    p.innerHTML = '<div style="font-size:13px;font-weight:600;margin-bottom:4px">View the dashboards as</div>' +
      '<div style="font-size:11px;color:#6b7280;margin-bottom:8px">Synthetic company. Each role sees only its own teams and tabs, as in production.</div>' +
      '<div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:12px">' + ROLES.map(function (r) {
        var on = _session && _session.role === r.value;
        return '<button type="button" data-role="' + r.value + '" style="font:600 11.5px inherit;padding:5px 9px;border-radius:14px;cursor:pointer;border:1px solid ' +
          (on ? '#1d4ed8;background:#1d4ed8;color:#fff' : '#d1d5db;background:#fff;color:#111827') + '">' + r.label + '</button>';
      }).join('') + '</div>' +
      '<div style="font-size:12px;font-weight:600;margin-bottom:4px">PINs for the locked tabs inside the reports</div>' + pinTable();
    p.addEventListener('click', function (e) {
      var t = e.target.closest ? e.target.closest('button[data-role]') : null; if (!t) return;
      var r = ROLES.filter(function (x) { return x.value === t.getAttribute('data-role'); })[0];
      if (r) { _save({ token: 'demo', role: r.value, team: r.team, label: r.label }); location.reload(); }
    });
    b.onclick = function () { p.style.display = p.style.display === 'none' ? 'block' : 'none'; };
    document.body.appendChild(p);
    document.body.appendChild(b);
  }

  function _renderLoginUI() {
    var overlay = document.createElement('div');
    overlay.id = 'northwind-auth-overlay';
    overlay.style.cssText = 'position:fixed;top:0;left:0;width:100vw;height:100vh;background:#f8f9fa;z-index:9999;display:flex;align-items:center;justify-content:center;font-family:"DM Sans",sans-serif';
    var pin = function (i) { return '<input type="password" maxlength="1" inputmode="numeric" id="auth-p' + i + '" style="width:52px;height:52px;border:1px solid #d1d5db;border-radius:8px;font-size:22px;font-weight:500;text-align:center;background:#fff;color:#111827">'; };
    overlay.style.overflowY = 'auto';
    overlay.innerHTML = '<div style="background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:28px 40px 40px;width:400px;max-width:calc(100vw - 32px);text-align:center;box-shadow:0 4px 24px rgba(0,0,0,.08);margin:24px 0">' +
      '<button id="auth-pin-toggle" type="button" style="width:100%;margin-bottom:18px;padding:9px 12px;background:#eaf1fc;color:#1d4ed8;border:1px solid #cfdcf6;border-radius:8px;cursor:pointer;font-size:12.5px;font-weight:600">' +
      '&#128273; Demo PINs – click to see which PIN is which</button>' +
      '<div id="auth-pin-list" style="display:none;margin:-8px 0 18px;padding:10px;border:1px solid #e5e7eb;border-radius:8px;background:#fafbfc">' + pinTable() + '</div>' +
      '<div style="font-size:13px;font-weight:500;color:#6b7280;margin-bottom:2px">Northwind Climate Co.</div>' +
      '<div style="font-size:20px;font-weight:500;color:#111827;margin-bottom:4px">ASAP Sales Dashboard</div>' +
      '<div style="font-size:11px;color:#9ca3af;margin-bottom:24px">Sign in to continue</div>' +
      '<div style="text-align:left;margin-bottom:12px"><label style="font-size:11px;font-weight:500;color:#374151;display:block;margin-bottom:4px">Your role</label>' +
      '<select id="auth-role-sel" style="width:100%;padding:8px 10px;border:1px solid #d1d5db;border-radius:8px;font-size:12px;color:#111827;background:#fff;height:36px">' +
      ROLES.map(function (r) { return '<option value="' + r.value + '">' + r.label + '</option>'; }).join('') + '</select></div>' +
      '<div style="text-align:left;margin-bottom:16px"><label style="font-size:11px;font-weight:500;color:#374151;display:block;margin-bottom:4px">PIN</label>' +
      '<div style="display:flex;gap:8px;justify-content:center">' + pin(0) + pin(1) + pin(2) + pin(3) + '</div></div>' +
      '<button id="auth-submit" style="width:100%;padding:10px;background:#1d4ed8;color:#fff;border:none;border-radius:8px;cursor:pointer;font-size:13px;font-weight:500;margin-bottom:8px">Sign in</button>' +
      '<div id="auth-error" style="font-size:11px;color:#dc2626;min-height:16px"></div>' +
      '<div style="font-size:10px;color:#9ca3af;margin-top:12px">Synthetic demo data &nbsp;&middot;&nbsp; Session ends when the tab closes</div></div>';
    document.body.appendChild(overlay);
    [0, 1, 2, 3].forEach(function (i) {
      var el = document.getElementById('auth-p' + i);
      el.addEventListener('input', function () { if (el.value && i < 3) document.getElementById('auth-p' + (i + 1)).focus(); if (i === 3 && el.value) _submit(); });
      el.addEventListener('keydown', function (e) { if (e.key === 'Backspace' && !el.value && i > 0) document.getElementById('auth-p' + (i - 1)).focus(); });
    });
    document.getElementById('auth-submit').addEventListener('click', _submit);
    document.getElementById('auth-pin-toggle').addEventListener('click', function () {
      var l = document.getElementById('auth-pin-list'); l.style.display = l.style.display === 'none' ? 'block' : 'none';
    });
    setTimeout(function () { var f = document.getElementById('auth-p0'); if (f) f.focus(); }, 100);
  }

  function _submit() {
    var role = document.getElementById('auth-role-sel').value;
    var pin = [0, 1, 2, 3].map(function (i) { return (document.getElementById('auth-p' + i) || { value: '' }).value; }).join('');
    var err = document.getElementById('auth-error');
    if (pin.length < 4) { err.textContent = 'Enter your 4-digit PIN'; return; }
    var r = ROLES.filter(function (x) { return x.value === role; })[0];
    if (!r || r.pin !== pin) {
      err.textContent = 'Incorrect PIN. Try again.';
      [0, 1, 2, 3].forEach(function (i) { document.getElementById('auth-p' + i).value = ''; });
      document.getElementById('auth-p0').focus();
      return;
    }
    _save({ token: 'demo', role: r.value, team: r.team, label: r.label });
    var overlay = document.getElementById('northwind-auth-overlay');
    if (r.value === 'evp' && overlay) {
      overlay.innerHTML = '<div style="text-align:center;font-family:\'DM Sans\',sans-serif"><div style="font-size:28px;margin-bottom:12px">&#128075;</div>' +
        '<div style="font-size:22px;font-weight:500;color:#111827;margin-bottom:6px">Welcome</div><div style="font-size:13px;color:#6b7280">Loading your dashboard...</div></div>';
      setTimeout(function () { overlay.remove(); if (_onLogin) _onLogin(_session); }, 1200);
    } else {
      if (overlay) overlay.remove();
      if (_onLogin) _onLogin(_session);
    }
  }

  function init(opts) {
    opts = opts || {};
    _onLogin = opts.onLogin || null;
    var go = function () {
      _load();
      // public demo: no sign-in screen - opens as EVP (everything visible); 'View as' switches role
      if (!_session || !_session.role) { var r0 = ROLES[0]; _save({ token: 'demo', role: r0.value, team: r0.team, label: r0.label }); }
      helpButton();
      if (opts.noLogin) return;
      if (_onLogin) _onLogin(_session);
    };
    if (document.body) go(); else document.addEventListener('DOMContentLoaded', go);
  }

  function getSession() { return _session; }
  function getRole() { return _session ? _session.role : null; }
  function getEmail() { return null; }
  function isAdmin() { return !!_session && FULL_ACCESS.indexOf(_session.role) >= 0; }
  function getTeam() { if (!_session || FULL_ACCESS.indexOf(_session.role) >= 0 || _session.role === 'se') return null; return _session.team || null; }
  function getTeams() { if (!_session || FULL_ACCESS.indexOf(_session.role) >= 0 || _session.role === 'se') return null; var t = _session.team; return Array.isArray(t) ? t : (t ? [t] : null); }
  function canSee(tab) { if (!_session) return false; var a = TAB_ACCESS[tab]; return !a || a.indexOf(_session.role) >= 0; }
  function logout() { var p = document.getElementById('demo-pin-help'); if (p) p.click(); }   // demo: 'Sign out' opens the role switcher
  var CL = {};
  function loadClaims(cb) { cb(CL); }
  function saveClaim(no, by, d, t, cb) { CL[no] = { claimed_by: by, claimed_date: d, claimed_team: t }; if (cb) cb({ ok: true }); }
  function deleteClaim(no, cb) { delete CL[no]; if (cb) cb({ ok: true }); }
  return { init: init, getSession: getSession, getRole: getRole, getEmail: getEmail, getTeam: getTeam, getTeams: getTeams,
           isAdmin: isAdmin, canSee: canSee, logout: logout, loadClaims: loadClaims, saveClaim: saveClaim, deleteClaim: deleteClaim,
           pins: ROLES };
})();
